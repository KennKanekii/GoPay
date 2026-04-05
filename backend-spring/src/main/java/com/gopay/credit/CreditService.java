package com.gopay.credit;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.gopay.auth.AuthService;
import com.gopay.auth.AuthService.StoredUser;
import com.gopay.transaction.TransactionService;
import com.gopay.transaction.TransactionService.Transaction;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.springframework.stereotype.Service;

@Service
public class CreditService {

  private static final String PYTHON_SCORE_URL = "http://localhost:5001/score";
  private static final int    HTTP_TIMEOUT_MS  = 3_000;

  private final AuthService        authService;
  private final TransactionService txService;
  private final ObjectMapper       objectMapper;

  public CreditService(AuthService authService,
                       TransactionService txService,
                       ObjectMapper objectMapper) {
    this.authService  = authService;
    this.txService    = txService;
    this.objectMapper = objectMapper;
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  public CreditController.CreditScoreResponse getScore(String authHeader) {
    StoredUser user = authService.getUserByToken(authHeader);
    Features   feat = computeFeatures(user);

    // Try the Python ML service first; fall back to rules if it is unreachable.
    try {
      return callPythonService(feat, user);
    } catch (Exception e) {
      return fallbackRuleScore(feat, user);
    }
  }

  // ---------------------------------------------------------------------------
  // Feature engineering — derived from real platform data
  // ---------------------------------------------------------------------------

  private Features computeFeatures(StoredUser user) {
    List<Transaction> txns = txService.getRawTransactionsForUser(user.id);

    double totalSent     = 0;
    double totalReceived = 0;
    String lastTxnAt     = null;

    for (Transaction t : txns) {
      if (Objects.equals(t.fromUserId, user.id)) totalSent     += t.amount;
      if (Objects.equals(t.toUserId,   user.id)) totalReceived += t.amount;
      if (lastTxnAt == null || t.createdAt.compareTo(lastTxnAt) > 0) lastTxnAt = t.createdAt;
    }

    long accountAgeDays     = accountAgeDays(user.createdAt);
    long daysSinceLastTxn   = lastTxnAt != null
        ? Math.max(0, ChronoUnit.DAYS.between(Instant.parse(lastTxnAt), Instant.now()))
        : accountAgeDays;

    double avgTxnAmount      = txns.isEmpty() ? 0 : (totalSent + totalReceived) / txns.size();
    double weeksAlive        = Math.max(1, accountAgeDays / 7.0);
    double txnFreqPerWeek    = txns.size() / weeksAlive;
    double balance           = user.balance > 0 ? user.balance : AuthService.DEFAULT_BALANCE;

    Features f = new Features();
    f.walletBalance          = balance;
    f.totalTransactions      = txns.size();
    f.totalSent              = totalSent;
    f.totalReceived          = totalReceived;
    f.avgTransactionAmount   = avgTxnAmount;
    f.accountAgeDays         = accountAgeDays;
    f.daysSinceLastTxn       = daysSinceLastTxn;
    f.txnFrequencyPerWeek    = txnFreqPerWeek;
    return f;
  }

  // ---------------------------------------------------------------------------
  // ML service call
  // ---------------------------------------------------------------------------

  @SuppressWarnings("unchecked")
  private CreditController.CreditScoreResponse callPythonService(Features f, StoredUser user)
      throws Exception {
    Map<String, Object> body = featureMap(f);
    byte[] payload = objectMapper.writeValueAsBytes(body);

    URL               url  = new URL(PYTHON_SCORE_URL);
    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
    conn.setRequestMethod("POST");
    conn.setRequestProperty("Content-Type", "application/json");
    conn.setDoOutput(true);
    conn.setConnectTimeout(HTTP_TIMEOUT_MS);
    conn.setReadTimeout(HTTP_TIMEOUT_MS);

    try (OutputStream os = conn.getOutputStream()) {
      os.write(payload);
    }

    int status = conn.getResponseCode();
    if (status != 200) {
      throw new RuntimeException("Python service returned HTTP " + status);
    }

    String raw = readStream(conn.getInputStream());
    Map<String, Object> json = objectMapper.readValue(raw, Map.class);

    CreditController.CreditScoreResponse r = new CreditController.CreditScoreResponse();
    r.userId      = user.id;
    r.name        = user.name;
    r.score       = ((Number) json.get("score")).intValue();
    r.riskBand    = (String) json.get("riskBand");
    r.colour      = (String) json.get("colour");
    r.tip         = (String) json.get("tip");
    r.model       = (String) json.get("model");
    r.breakdown   = (Map<String, Object>) json.get("breakdown");
    r.features    = featureMap(f);
    return r;
  }

  // ---------------------------------------------------------------------------
  // Rule-based fallback (no Python dependency)
  // ---------------------------------------------------------------------------

  private CreditController.CreditScoreResponse fallbackRuleScore(Features f, StoredUser user) {
    double bScore  = Math.min(1.0, f.walletBalance / 75_000.0);
    double aScore  = Math.min(1.0, f.totalTransactions / 200.0);
    double total   = f.totalSent + f.totalReceived;
    double nfScore = total > 0 ? f.totalReceived / total : 0.5;
    double rScore  = f.totalTransactions > 0
        ? Math.max(0, 1.0 - f.daysSinceLastTxn / 60.0) : 0.0;
    double ageScore = Math.min(1.0, f.accountAgeDays / 1095.0);

    double raw   = bScore * 0.30 + aScore * 0.25 + nfScore * 0.25 + rScore * 0.10 + ageScore * 0.10;
    int    score = (int) Math.round(300 + raw * 600);
    score = Math.max(300, Math.min(900, score));

    Map<String, Object> breakdown = new HashMap<>();
    breakdown.put("balanceFactor",    round1(bScore   * 100));
    breakdown.put("activityFactor",   round1(aScore   * 100));
    breakdown.put("netFlowFactor",    round1(nfScore  * 100));
    breakdown.put("recencyFactor",    round1(rScore   * 100));
    breakdown.put("accountAgeFactor", round1(ageScore * 100));

    CreditController.CreditScoreResponse r = new CreditController.CreditScoreResponse();
    r.userId    = user.id;
    r.name      = user.name;
    r.score     = score;
    r.riskBand  = bandFor(score);
    r.colour    = colourFor(score);
    r.tip       = tipFor(score);
    r.model     = "rule_based_fallback";
    r.breakdown = breakdown;
    r.features  = featureMap(f);
    return r;
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  private Map<String, Object> featureMap(Features f) {
    Map<String, Object> m = new HashMap<>();
    m.put("wallet_balance",          f.walletBalance);
    m.put("total_transactions",      f.totalTransactions);
    m.put("total_sent",              f.totalSent);
    m.put("total_received",          f.totalReceived);
    m.put("avg_transaction_amount",  f.avgTransactionAmount);
    m.put("account_age_days",        f.accountAgeDays);
    m.put("days_since_last_txn",     f.daysSinceLastTxn);
    m.put("txn_frequency_per_week",  f.txnFrequencyPerWeek);
    return m;
  }

  private static long accountAgeDays(String createdAt) {
    if (createdAt == null || createdAt.isEmpty()) return 1;
    try {
      return Math.max(1, ChronoUnit.DAYS.between(Instant.parse(createdAt), Instant.now()));
    } catch (Exception e) {
      return 1;
    }
  }

  private static double round1(double v) {
    return Math.round(v * 10.0) / 10.0;
  }

  private static String readStream(InputStream is) throws Exception {
    byte[] buffer = new byte[4096];
    StringBuilder sb = new StringBuilder();
    int read;
    while ((read = is.read(buffer)) != -1) {
      sb.append(new String(buffer, 0, read, StandardCharsets.UTF_8));
    }
    return sb.toString();
  }

  private static String bandFor(int score) {
    if (score >= 800) return "EXCELLENT";
    if (score >= 740) return "VERY_GOOD";
    if (score >= 670) return "GOOD";
    if (score >= 580) return "FAIR";
    return "POOR";
  }

  private static String colourFor(int score) {
    if (score >= 800) return "#16a34a";
    if (score >= 740) return "#4ade80";
    if (score >= 670) return "#86efac";
    if (score >= 580) return "#fbbf24";
    return "#ef4444";
  }

  private static String tipFor(int score) {
    if (score >= 800) return "Exceptional credit health. Eligible for the best loan rates.";
    if (score >= 740) return "Very strong credit. Most lenders will offer favourable terms.";
    if (score >= 670) return "Good credit standing. Eligible for most standard loan products.";
    if (score >= 580) return "Fair credit. Some lenders may require higher interest rates.";
    return "Poor credit. Increase your balance and transaction activity to improve your score.";
  }

  // ---------------------------------------------------------------------------
  // Feature POJO
  // ---------------------------------------------------------------------------

  private static class Features {
    double walletBalance;
    int    totalTransactions;
    double totalSent;
    double totalReceived;
    double avgTransactionAmount;
    long   accountAgeDays;
    long   daysSinceLastTxn;
    double txnFrequencyPerWeek;
  }
}
