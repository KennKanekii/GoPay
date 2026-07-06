package com.gopay.fraud;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.gopay.auth.AuthService;
import com.gopay.auth.AuthService.StoredUser;
import com.gopay.transaction.TransactionService.Transaction;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import org.springframework.stereotype.Service;

/**
 * GoPay Fraud Risk Scorer
 * ========================
 * Replicates the multi-layer fraud detection used by leading fintech companies:
 *
 *   Layer 1 — Blacklist Check      : instant BLOCK on known bad actors
 *   Layer 2 — Velocity Rules       : hard limits on transaction rate/amount
 *   Layer 3 — Behavioral Analysis  : deviations from the user's own history
 *   Layer 4 — ML Scoring           : Python RandomForest model via REST
 *   Layer 5 — Rule Fallback        : deterministic score if ML is unreachable
 *
 * Every transaction is assessed before funds move, and the result is stored
 * with the transaction for audit / compliance.
 */
@Service
public class FraudService {

  // -------------------------------------------------------------------------
  // Configuration (in a real system these come from a config server / DB)
  // -------------------------------------------------------------------------
  private static final String FRAUD_ML_URL        = "http://localhost:5002/assess";
  private static final int    HTTP_TIMEOUT_MS      = 3_000;

  /** Velocity limits — aligned with RBI guidelines and UPI velocity controls */
  private static final int    MAX_TXNS_PER_1H      = 5;
  private static final double MAX_AMOUNT_PER_1H    = 20_000.0;   // Rs. 20k/hr
  private static final int    MAX_TXNS_PER_24H     = 20;
  private static final double MAX_AMOUNT_PER_24H   = 1_00_000.0; // Rs. 1L/day
  private static final int    MAX_RECIPIENTS_24H   = 8;

  /** Behavioural thresholds */
  private static final double NEW_RECIP_LIMIT      = 10_000.0;   // Rs.10k to new recip
  private static final double BALANCE_DRAIN_RATIO  = 0.75;       // 75% of balance
  private static final double AMT_TO_AVG_RATIO     = 5.0;        // 5x historical avg

  // -------------------------------------------------------------------------
  // Dependencies
  // -------------------------------------------------------------------------
  private final AuthService  authService;
  private final ObjectMapper objectMapper;
  private final Path         transactionsFile;
  private final Path         fraudEventsFile;

  // Resolved blacklist (cached in memory — refreshed on startup)
  private final Set<String> blockedDomains   = new HashSet<>();
  private final Set<String> blockedEmails    = new HashSet<>();
  private final Set<String> blockedKeywords  = new HashSet<>();

  public FraudService(AuthService authService, ObjectMapper objectMapper) {
    this.authService      = authService;
    this.objectMapper     = objectMapper;
    this.transactionsFile = authService.getDataDir().resolve("transactions.json");
    this.fraudEventsFile  = authService.getDataDir().resolve("fraud_events.json");
    loadBlacklist();
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Assess fraud risk for a proposed transaction.
   * Called by TransactionService BEFORE funds are moved.
   *
   * @return FraudAssessment — contains recommendation (ALLOW/REVIEW/BLOCK),
   *         risk score 0-100, and list of triggered signals.
   */
  public FraudAssessment assess(StoredUser sender, StoredUser recipient,
                                 double amount, String note) {
    List<Transaction> recentSent = recentSentBy(sender.id);
    FraudFeatures feat = buildFeatures(sender, recipient, amount, recentSent);

    // --- Layer 0: VPA Spoofing + IFSC Fraud (hard signals, computed first) ---
    enrichWithVpaAndIfsc(feat, recipient);

    // Layer 0a: VPA spoofing detected — immediate BLOCK
    if (feat.vpaRiskScore >= 80) {
      FraudAssessment blocked = new FraudAssessment();
      blocked.fraudScore     = 90;
      blocked.riskLevel      = "CRITICAL";
      blocked.recommendation = "BLOCK";
      blocked.signals        = Arrays.asList("vpa_spoofing_detected");
      blocked.signalLabels   = Arrays.asList(
          "VPA spoofing detected: recipient handle closely resembles a legitimate UPI handle.");
      blocked.model          = "vpa_levenshtein";
      logEvent(sender, recipient, amount, blocked);
      return blocked;
    }

    // Layer 0b: Invalid IFSC + large amount — hard BLOCK
    if (feat.ifscIsValid == 0 && amount >= 5_000) {
      FraudAssessment blocked = new FraudAssessment();
      blocked.fraudScore     = 85;
      blocked.riskLevel      = "CRITICAL";
      blocked.recommendation = "BLOCK";
      blocked.signals        = Arrays.asList("invalid_ifsc");
      blocked.signalLabels   = Arrays.asList(
          "Transaction blocked: recipient IFSC code is invalid or unregistered.");
      blocked.model          = "ifsc_validator";
      logEvent(sender, recipient, amount, blocked);
      return blocked;
    }

    // --- Layer 1: Blacklist (hard block, no ML needed) ---
    if (feat.isBlacklisted == 1) {
      FraudAssessment blocked = new FraudAssessment();
      blocked.fraudScore  = 95;
      blocked.riskLevel   = "CRITICAL";
      blocked.recommendation = "BLOCK";
      blocked.signals     = Arrays.asList("blacklisted_recipient");
      blocked.signalLabels = Arrays.asList("Recipient is on the GoPay fraud blacklist.");
      blocked.model       = "blacklist";
      logEvent(sender, recipient, amount, blocked);
      return blocked;
    }

    // --- Layer 2: Hard velocity breaches (immediate BLOCK) ---
    FraudAssessment velocityResult = checkHardVelocity(feat, recentSent, amount);
    if (velocityResult != null) {
      logEvent(sender, recipient, amount, velocityResult);
      return velocityResult;
    }

    // --- Layers 3+4: ML model (with rule-based fallback) ---
    FraudAssessment assessment;
    try {
      assessment = callMlService(feat);
    } catch (Exception e) {
      assessment = ruleFallback(feat);
    }

    logEvent(sender, recipient, amount, assessment);
    return assessment;
  }

  /** Return the fraud event log for a specific user (sent and received). */
  public List<FraudEvent> getEventsForUser(String userId) {
    return readFraudEvents().stream()
        .filter(e -> Objects.equals(e.fromUserId, userId))
        .sorted((a, b) -> b.createdAt.compareTo(a.createdAt))
        .collect(Collectors.toList());
  }

  /** Velocity summary for the dashboard. */
  public VelocitySummary getVelocitySummary(StoredUser user) {
    List<Transaction> recentSent = recentSentBy(user.id);
    Instant now  = Instant.now();
    Instant h1   = now.minus(1,  ChronoUnit.HOURS);
    Instant h24  = now.minus(24, ChronoUnit.HOURS);

    long    txns1h  = recentSent.stream().filter(t -> parseInstant(t.createdAt).isAfter(h1)).count();
    long    txns24h = recentSent.stream().filter(t -> parseInstant(t.createdAt).isAfter(h24)).count();
    double  amt1h   = recentSent.stream().filter(t -> parseInstant(t.createdAt).isAfter(h1))
                                .mapToDouble(t -> t.amount).sum();
    double  amt24h  = recentSent.stream().filter(t -> parseInstant(t.createdAt).isAfter(h24))
                                .mapToDouble(t -> t.amount).sum();
    long    recips  = recentSent.stream().filter(t -> parseInstant(t.createdAt).isAfter(h24))
                                .map(t -> t.toUserId).distinct().count();

    VelocitySummary vs = new VelocitySummary();
    vs.txns1h          = (int) txns1h;
    vs.txns24h         = (int) txns24h;
    vs.amountSent1h    = amt1h;
    vs.amountSent24h   = amt24h;
    vs.uniqueRecips24h = (int) recips;
    vs.limitTxns1h     = MAX_TXNS_PER_1H;
    vs.limitAmount1h   = MAX_AMOUNT_PER_1H;
    vs.limitTxns24h    = MAX_TXNS_PER_24H;
    vs.limitAmount24h  = MAX_AMOUNT_PER_24H;
    vs.limitRecips24h  = MAX_RECIPIENTS_24H;
    return vs;
  }

  // -------------------------------------------------------------------------
  // Feature engineering — same feature set as Python model
  // -------------------------------------------------------------------------

  private FraudFeatures buildFeatures(StoredUser sender, StoredUser recipient,
                                       double amount, List<Transaction> recentSent) {
    Instant now = Instant.now();
    Instant h1  = now.minus(1,  ChronoUnit.HOURS);
    Instant h24 = now.minus(24, ChronoUnit.HOURS);

    List<Transaction> sent1h  = recentSent.stream()
        .filter(t -> parseInstant(t.createdAt).isAfter(h1)).collect(Collectors.toList());
    List<Transaction> sent24h = recentSent.stream()
        .filter(t -> parseInstant(t.createdAt).isAfter(h24)).collect(Collectors.toList());

    double balance = sender.balance > 0 ? sender.balance : AuthService.DEFAULT_BALANCE;

    // Historical average amount (all-time sent by this user)
    double histAvg = recentSent.isEmpty() ? amount
        : recentSent.stream().mapToDouble(t -> t.amount).average().orElse(amount);

    // Has this sender sent to this recipient before?
    boolean knownRecipient = recentSent.stream()
        .anyMatch(t -> Objects.equals(t.toUserId, recipient.id));

    long acctAgeDays = ChronoUnit.DAYS.between(
        parseInstant(sender.createdAt != null ? sender.createdAt : now.toString()), now);

    int hour      = now.atZone(java.time.ZoneId.of("Asia/Kolkata")).getHour();
    int dayOfWeek = now.atZone(java.time.ZoneId.of("Asia/Kolkata")).getDayOfWeek().getValue();

    long uniqueRecipients24h = sent24h.stream().map(t -> t.toUserId).distinct().count();

    // Blacklist check
    boolean blacklisted = isBlacklisted(recipient.identifier);

    // Amount z-score: (amount - mean_sent) / std_sent
    double meanSent = recentSent.isEmpty() ? amount
        : recentSent.stream().mapToDouble(t -> t.amount).average().orElse(amount);
    double stdSent  = 1.0;
    if (recentSent.size() > 1) {
      double sumSq = recentSent.stream().mapToDouble(t -> Math.pow(t.amount - meanSent, 2)).sum();
      stdSent = Math.sqrt(sumSq / recentSent.size());
    }
    double amountZscore = (amount - meanSent) / Math.max(stdSent, 1.0);

    FraudFeatures f = new FraudFeatures();
    f.amount                = amount;
    f.amountToBalanceRatio  = amount / Math.max(balance, 1);
    f.amountToAvgRatio      = amount / Math.max(histAvg, 1);
    f.txnsLast1h            = sent1h.size();
    f.txnsLast24h           = sent24h.size();
    f.amountSentLast1h      = sent1h.stream().mapToDouble(t -> t.amount).sum();
    f.amountSentLast24h     = sent24h.stream().mapToDouble(t -> t.amount).sum();
    f.uniqueRecipients24h   = (int) uniqueRecipients24h;
    f.isNewRecipient        = knownRecipient ? 0 : 1;
    f.hourOfDay             = hour;
    f.isNight               = (hour < 6) ? 1 : 0;
    f.isWeekend             = (dayOfWeek >= 6) ? 1 : 0;
    f.accountAgeDays        = (int) Math.max(1, acctAgeDays);
    f.isRoundAmount         = (amount > 500 && amount % 1000 < 1) ? 1 : 0;
    f.isBlacklisted         = blacklisted ? 1 : 0;
    f.balanceAfterRatio     = Math.max(0, balance - amount) / Math.max(balance, 1);
    f.amountZscore          = amountZscore;
    f.senderHasVpa          = (sender.vpa != null && !sender.vpa.isEmpty()) ? 1 : 0;
    // vpaRiskScore and ifscIsValid are populated by enrichWithVpaAndIfsc() after this
    f.vpaRiskScore          = 0;
    f.ifscIsValid           = 1; // default: assume valid until checked
    return f;
  }

  /**
   * Layer 0 enrichment: calls Python /vpa-check and /ifsc-validate
   * to compute vpaRiskScore and ifscIsValid for the recipient.
   * Falls back to safe defaults if the Python service is unavailable.
   */
  @SuppressWarnings("unchecked")
  private void enrichWithVpaAndIfsc(FraudFeatures feat, StoredUser recipient) {
    // --- VPA check ---
    try {
      String recipientVpa = recipient.vpa;
      if (recipientVpa != null && !recipientVpa.isEmpty()) {
        Map<String, Object> vpaBody = new HashMap<>();
        vpaBody.put("vpa", recipientVpa);
        byte[]            vpaPayload = objectMapper.writeValueAsBytes(vpaBody);
        URL               vpaUrl     = new URL("http://localhost:5002/vpa-check");
        HttpURLConnection vpaConn    = (HttpURLConnection) vpaUrl.openConnection();
        vpaConn.setRequestMethod("POST");
        vpaConn.setRequestProperty("Content-Type", "application/json");
        vpaConn.setDoOutput(true);
        vpaConn.setConnectTimeout(1_000);
        vpaConn.setReadTimeout(1_000);
        try (OutputStream os = vpaConn.getOutputStream()) { os.write(vpaPayload); }
        if (vpaConn.getResponseCode() == 200) {
          byte[] resp = readStream(vpaConn.getInputStream());
          Map<String, Object> json = objectMapper.readValue(resp, Map.class);
          Object rs = json.get("risk_score");
          if (rs instanceof Number) feat.vpaRiskScore = ((Number) rs).intValue();
        }
      }
    } catch (Exception e) {
      // VPA service unavailable — use 0 (safe default)
    }

    // --- IFSC check ---
    try {
      String ifsc = recipient.ifscCode;
      if (ifsc != null && !ifsc.isEmpty()) {
        Map<String, Object> ifscBody = new HashMap<>();
        ifscBody.put("ifsc", ifsc);
        byte[]            ifscPayload = objectMapper.writeValueAsBytes(ifscBody);
        URL               ifscUrl     = new URL("http://localhost:5002/ifsc-validate");
        HttpURLConnection ifscConn    = (HttpURLConnection) ifscUrl.openConnection();
        ifscConn.setRequestMethod("POST");
        ifscConn.setRequestProperty("Content-Type", "application/json");
        ifscConn.setDoOutput(true);
        ifscConn.setConnectTimeout(1_000);
        ifscConn.setReadTimeout(1_000);
        try (OutputStream os = ifscConn.getOutputStream()) { os.write(ifscPayload); }
        if (ifscConn.getResponseCode() == 200) {
          byte[] resp = readStream(ifscConn.getInputStream());
          Map<String, Object> json = objectMapper.readValue(resp, Map.class);
          Object valid = json.get("is_valid");
          if (valid instanceof Boolean) feat.ifscIsValid = ((Boolean) valid) ? 1 : 0;
        }
      }
    } catch (Exception e) {
      // IFSC service unavailable — default to valid (1)
    }
  }

  // -------------------------------------------------------------------------
  // Layer 2 — Hard velocity breaches (deterministic, immediate BLOCK)
  // -------------------------------------------------------------------------

  private FraudAssessment checkHardVelocity(FraudFeatures feat,
                                              List<Transaction> recentSent,
                                              double amount) {
    List<String> signals = new ArrayList<>();
    List<String> labels  = new ArrayList<>();

    if (feat.txnsLast1h >= MAX_TXNS_PER_1H) {
      signals.add("velocity_limit_1h_count");
      labels.add(String.format("Exceeded %d transactions/hour limit (%d attempted).",
          MAX_TXNS_PER_1H, feat.txnsLast1h + 1));
    }
    if (feat.amountSentLast1h + amount > MAX_AMOUNT_PER_1H) {
      signals.add("velocity_limit_1h_amount");
      labels.add(String.format("Hourly spend limit Rs.%.0f exceeded.", MAX_AMOUNT_PER_1H));
    }
    if (feat.txnsLast24h >= MAX_TXNS_PER_24H) {
      signals.add("velocity_limit_24h_count");
      labels.add(String.format("Exceeded %d transactions/day limit.", MAX_TXNS_PER_24H));
    }
    if (feat.amountSentLast24h + amount > MAX_AMOUNT_PER_24H) {
      signals.add("velocity_limit_24h_amount");
      labels.add(String.format("Daily spend limit Rs.%.0f exceeded.", MAX_AMOUNT_PER_24H));
    }

    if (signals.isEmpty()) return null;

    FraudAssessment a = new FraudAssessment();
    a.fraudScore     = 90;
    a.riskLevel      = "CRITICAL";
    a.recommendation = "BLOCK";
    a.signals        = signals;
    a.signalLabels   = labels;
    a.model          = "velocity_rules";
    return a;
  }

  // -------------------------------------------------------------------------
  // Layer 4 — ML service call
  // -------------------------------------------------------------------------

  @SuppressWarnings("unchecked")
  private FraudAssessment callMlService(FraudFeatures feat) throws Exception {
    Map<String, Object> body = featureMap(feat);
    byte[] payload = objectMapper.writeValueAsBytes(body);

    URL conn = new URL(FRAUD_ML_URL);
    HttpURLConnection http = (HttpURLConnection) conn.openConnection();
    http.setRequestMethod("POST");
    http.setRequestProperty("Content-Type", "application/json");
    http.setDoOutput(true);
    http.setConnectTimeout(HTTP_TIMEOUT_MS);
    http.setReadTimeout(HTTP_TIMEOUT_MS);

    try (OutputStream os = http.getOutputStream()) { os.write(payload); }

    if (http.getResponseCode() != 200) {
      throw new RuntimeException("ML service HTTP " + http.getResponseCode());
    }

    byte[] resp = readStream(http.getInputStream());
    Map<String, Object> json = objectMapper.readValue(resp, Map.class);

    List<Map<String, Object>> rawSignals = (List<Map<String, Object>>) json.getOrDefault("signals", new ArrayList<>());
    List<String> codes  = new ArrayList<>();
    List<String> labels = new ArrayList<>();
    for (Map<String, Object> s : rawSignals) {
      codes.add((String) s.getOrDefault("code", ""));
      labels.add((String) s.getOrDefault("label", ""));
    }

    FraudAssessment a = new FraudAssessment();
    a.fraudScore     = ((Number) json.get("fraudScore")).intValue();
    a.riskLevel      = (String) json.get("riskLevel");
    a.recommendation = (String) json.get("recommendation");
    a.signals        = codes;
    a.signalLabels   = labels;
    a.model          = "ml_random_forest";
    return a;
  }

  // -------------------------------------------------------------------------
  // Layer 5 — Rule-based fallback
  // -------------------------------------------------------------------------

  private FraudAssessment ruleFallback(FraudFeatures feat) {
    int score = 0;
    List<String> signals = new ArrayList<>();
    List<String> labels  = new ArrayList<>();

    // Velocity score
    double velScore = Math.min(1.0,
        (double) feat.txnsLast1h / MAX_TXNS_PER_1H * 0.4 +
        feat.amountSentLast1h   / MAX_AMOUNT_PER_1H * 0.3 +
        (double) feat.txnsLast24h / MAX_TXNS_PER_24H * 0.3);
    score += (int) (velScore * 35);

    // Behavioural score
    if (feat.amountToBalanceRatio >= BALANCE_DRAIN_RATIO) {
      score += 20;
      signals.add("high_balance_drain");
      labels.add(String.format("Transaction drains %.0f%% of balance.", feat.amountToBalanceRatio * 100));
    }
    if (feat.amountToAvgRatio >= AMT_TO_AVG_RATIO) {
      score += 15;
      signals.add("unusual_amount");
      labels.add(String.format("Amount is %.1fx the historical average.", feat.amountToAvgRatio));
    }
    if (feat.isNewRecipient == 1 && feat.amount >= NEW_RECIP_LIMIT) {
      score += 15;
      signals.add("large_new_recipient");
      labels.add("Large amount to a first-time recipient.");
    }
    if (feat.isNight == 1 && feat.isNewRecipient == 1) {
      score += 10;
      signals.add("night_new_recipient");
      labels.add("Late-night transaction to new recipient.");
    }
    if (feat.uniqueRecipients24h >= MAX_RECIPIENTS_24H) {
      score += 10;
      signals.add("many_recipients");
      labels.add("Too many distinct recipients in 24 hours.");
    }
    if (feat.accountAgeDays < 7 && feat.amount >= 5_000) {
      score += 10;
      signals.add("new_account_large_txn");
      labels.add("New account (<7 days) making a large transaction.");
    }

    score = Math.min(100, score);
    String riskLevel = score >= 80 ? "CRITICAL" : score >= 60 ? "HIGH" : score >= 35 ? "MEDIUM" : "LOW";
    String rec       = score >= 80 ? "BLOCK" : "ALLOW";

    FraudAssessment a = new FraudAssessment();
    a.fraudScore     = score;
    a.riskLevel      = riskLevel;
    a.recommendation = rec;
    a.signals        = signals.isEmpty() ? Arrays.asList("low_risk") : signals;
    a.signalLabels   = labels.isEmpty()  ? Arrays.asList("No significant risk signals detected.") : labels;
    a.model          = "rule_based_fallback";
    return a;
  }

  // -------------------------------------------------------------------------
  // Blacklist
  // -------------------------------------------------------------------------

  @SuppressWarnings("unchecked")
  private void loadBlacklist() {
    // Look for blacklist.json relative to the project root (fraud-engine folder)
    Path[] candidates = {
        authService.getDataDir().resolve("../fraud-engine/blacklist.json").normalize(),
        authService.getDataDir().resolve("../../fraud-engine/blacklist.json").normalize(),
        java.nio.file.Paths.get("../fraud-engine/blacklist.json").toAbsolutePath().normalize(),
    };

    for (Path p : candidates) {
      if (Files.exists(p)) {
        try {
          String raw = new String(Files.readAllBytes(p), StandardCharsets.UTF_8);
          Map<String, Object> bl = objectMapper.readValue(raw, Map.class);
          blockedDomains.addAll((List<String>) bl.getOrDefault("blockedDomains", new ArrayList<>()));
          blockedEmails.addAll((List<String>) bl.getOrDefault("blockedEmails", new ArrayList<>()));
          blockedKeywords.addAll((List<String>) bl.getOrDefault("blockedKeywords", new ArrayList<>()));
          System.out.println("[FraudService] Loaded blacklist: "
              + blockedDomains.size() + " domains, "
              + blockedEmails.size() + " emails.");
          return;
        } catch (Exception e) {
          System.err.println("[FraudService] Could not parse blacklist: " + e.getMessage());
        }
      }
    }
    System.out.println("[FraudService] No blacklist.json found — blacklist checks disabled.");
  }

  private boolean isBlacklisted(String identifier) {
    if (identifier == null) return false;
    identifier = identifier.toLowerCase().trim();
    if (blockedEmails.contains(identifier)) return true;
    int at = identifier.indexOf('@');
    if (at >= 0) {
      String domain = identifier.substring(at + 1);
      if (blockedDomains.contains(domain)) return true;
    }
    for (String kw : blockedKeywords) {
      if (identifier.contains(kw)) return true;
    }
    return false;
  }

  // -------------------------------------------------------------------------
  // Fraud event log (audit trail)
  // -------------------------------------------------------------------------

  private void logEvent(StoredUser sender, StoredUser recipient,
                         double amount, FraudAssessment a) {
    try {
      List<FraudEvent> events = readFraudEvents();
      FraudEvent e = new FraudEvent();
      e.id             = "fe_" + System.currentTimeMillis();
      e.fromUserId     = sender.id;
      e.fromName       = sender.name;
      e.fromIdentifier = sender.identifier;
      e.toIdentifier   = recipient.identifier;
      e.amount         = amount;
      e.fraudScore     = a.fraudScore;
      e.riskLevel      = a.riskLevel;
      e.recommendation = a.recommendation;
      e.signals        = a.signals;
      e.model          = a.model;
      e.createdAt      = Instant.now().toString();
      events.add(e);
      // Keep only last 1000 events
      if (events.size() > 1000) events = events.subList(events.size() - 1000, events.size());
      objectMapper.writerWithDefaultPrettyPrinter()
          .writeValue(fraudEventsFile.toFile(), events);
    } catch (Exception ex) {
      System.err.println("[FraudService] Could not log fraud event: " + ex.getMessage());
    }
  }

  private List<FraudEvent> readFraudEvents() {
    try {
      if (!Files.exists(fraudEventsFile)) {
        Files.write(fraudEventsFile, "[]".getBytes(StandardCharsets.UTF_8));
      }
      String raw = new String(Files.readAllBytes(fraudEventsFile), StandardCharsets.UTF_8);
      List<FraudEvent> list = objectMapper.readValue(raw, new TypeReference<List<FraudEvent>>() {});
      return list != null ? list : new ArrayList<>();
    } catch (Exception e) {
      return new ArrayList<>();
    }
  }

  // -------------------------------------------------------------------------
  // Helper — recent sent transactions for velocity computation
  // -------------------------------------------------------------------------

  private List<Transaction> recentSentBy(String userId) {
    try {
      if (!Files.exists(transactionsFile)) return new ArrayList<>();
      String raw = new String(Files.readAllBytes(transactionsFile), StandardCharsets.UTF_8);
      List<Transaction> all = objectMapper.readValue(raw, new TypeReference<List<Transaction>>() {});
      if (all == null) return new ArrayList<>();
      Instant cutoff = Instant.now().minus(30, ChronoUnit.DAYS); // 30-day window
      return all.stream()
          .filter(t -> Objects.equals(t.fromUserId, userId)
              && parseInstant(t.createdAt).isAfter(cutoff))
          .collect(Collectors.toList());
    } catch (Exception e) {
      return new ArrayList<>();
    }
  }

  private static Instant parseInstant(String iso) {
    try { return Instant.parse(iso); } catch (Exception e) { return Instant.EPOCH; }
  }

  private Map<String, Object> featureMap(FraudFeatures f) {
    Map<String, Object> m = new HashMap<>();
    m.put("amount",                  f.amount);
    m.put("amount_to_balance_ratio", f.amountToBalanceRatio);
    m.put("amount_to_avg_ratio",     f.amountToAvgRatio);
    m.put("txns_last_1h",            f.txnsLast1h);
    m.put("txns_last_24h",           f.txnsLast24h);
    m.put("amount_sent_last_1h",     f.amountSentLast1h);
    m.put("amount_sent_last_24h",    f.amountSentLast24h);
    m.put("unique_recipients_24h",   f.uniqueRecipients24h);
    m.put("is_new_recipient",        f.isNewRecipient);
    m.put("hour_of_day",             f.hourOfDay);
    m.put("is_night",                f.isNight);
    m.put("is_weekend",              f.isWeekend);
    m.put("account_age_days",        f.accountAgeDays);
    m.put("is_round_amount",         f.isRoundAmount);
    m.put("is_blacklisted",          f.isBlacklisted);
    m.put("balance_after_ratio",     f.balanceAfterRatio);
    m.put("vpa_risk_score",          f.vpaRiskScore);
    m.put("ifsc_is_valid",           f.ifscIsValid);
    m.put("amount_zscore",           f.amountZscore);
    m.put("sender_has_vpa",          f.senderHasVpa);
    return m;
  }

  private static byte[] readStream(InputStream is) throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    byte[] buf = new byte[4096];
    int n;
    while ((n = is.read(buf)) != -1) baos.write(buf, 0, n);
    return baos.toByteArray();
  }

  // -------------------------------------------------------------------------
  // Public DTOs
  // -------------------------------------------------------------------------

  public static class FraudAssessment {
    public int          fraudScore;      // 0-100
    public String       riskLevel;       // LOW / MEDIUM / HIGH / CRITICAL
    public String       recommendation;  // ALLOW / REVIEW / BLOCK
    public List<String> signals;         // signal codes
    public List<String> signalLabels;    // human-readable descriptions
    public String       model;           // which layer made the decision
  }

  public static class FraudFeatures {
    public double amount;
    public double amountToBalanceRatio;
    public double amountToAvgRatio;
    public int    txnsLast1h;
    public int    txnsLast24h;
    public double amountSentLast1h;
    public double amountSentLast24h;
    public int    uniqueRecipients24h;
    public int    isNewRecipient;
    public int    hourOfDay;
    public int    isNight;
    public int    isWeekend;
    public int    accountAgeDays;
    public int    isRoundAmount;
    public int    isBlacklisted;
    public double balanceAfterRatio;
    // v2 additions
    public int    vpaRiskScore;    // 0-100 from Levenshtein VPA check
    public int    ifscIsValid;     // 1 = valid, 0 = invalid/unknown
    public double amountZscore;    // z-score of amount vs sender history
    public int    senderHasVpa;    // 1 if sender registered a UPI VPA
  }

  public static class FraudEvent {
    public String       id;
    public String       fromUserId;
    public String       fromName;
    public String       fromIdentifier;
    public String       toIdentifier;
    public double       amount;
    public int          fraudScore;
    public String       riskLevel;
    public String       recommendation;
    public List<String> signals;
    public String       model;
    public String       createdAt;
  }

  public static class VelocitySummary {
    public int    txns1h;
    public int    txns24h;
    public double amountSent1h;
    public double amountSent24h;
    public int    uniqueRecips24h;
    public int    limitTxns1h;
    public double limitAmount1h;
    public int    limitTxns24h;
    public double limitAmount24h;
    public int    limitRecips24h;
  }
}
