package com.gopay.transaction;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.gopay.auth.AuthService;
import com.gopay.auth.AuthService.StoredUser;
import com.gopay.fraud.FraudService;
import com.gopay.fraud.FraudService.FraudAssessment;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.SecureRandom;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import org.springframework.stereotype.Service;

@Service
public class TransactionService {

  private final AuthService   authService;
  private final FraudService  fraudService;
  private final ObjectMapper  objectMapper;
  private final Path          transactionsFile;

  private static final SecureRandom RANDOM = new SecureRandom();

  public TransactionService(AuthService authService,
                             FraudService fraudService,
                             ObjectMapper objectMapper) {
    this.authService      = authService;
    this.fraudService     = fraudService;
    this.objectMapper     = objectMapper;
    this.transactionsFile = authService.getDataDir().resolve("transactions.json");
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  public TransactionController.BalanceResponse getBalance(String authHeader) {
    StoredUser user    = authService.getUserByToken(authHeader);
    double     balance = user.balance > 0 ? user.balance : AuthService.DEFAULT_BALANCE;

    TransactionController.BalanceResponse resp = new TransactionController.BalanceResponse();
    resp.balance  = balance;
    resp.currency = "INR";
    return resp;
  }

  public TransactionController.SendResponse send(String authHeader,
      TransactionController.SendRequest body) {

    StoredUser sender = authService.getUserByToken(authHeader);

    // --- Validate request body ---
    String recipientId = body == null ? "" : (body.recipientIdentifier == null ? "" : body.recipientIdentifier.trim());
    if (recipientId.isEmpty()) {
      throw new AuthService.BadRequestException("Recipient email/phone is required.");
    }

    double amount = body == null ? 0 : body.amount;
    if (amount <= 0) {
      throw new AuthService.BadRequestException("Amount must be greater than Rs.0.");
    }
    if (amount > 1_00_000) {
      throw new AuthService.BadRequestException("Amount cannot exceed Rs.1,00,000 per transaction.");
    }

    String note = (body == null || body.note == null) ? "" : body.note.trim();

    // --- Find recipient ---
    String          recipientIdentifier = recipientId.toLowerCase();
    List<StoredUser> users              = authService.readUsers();
    StoredUser recipient = users.stream()
        .filter(u -> Objects.equals(u.identifier, recipientIdentifier))
        .findFirst().orElse(null);

    if (recipient == null) {
      throw new AuthService.BadRequestException("No GoPay account found for: " + recipientId);
    }
    if (Objects.equals(sender.id, recipient.id)) {
      throw new AuthService.BadRequestException("You cannot send money to yourself.");
    }

    // --- Balance check ---
    double senderBalance    = sender.balance > 0 ? sender.balance : AuthService.DEFAULT_BALANCE;
    double recipientBalance = recipient.balance > 0 ? recipient.balance : AuthService.DEFAULT_BALANCE;
    if (senderBalance < amount) {
      throw new AuthService.BadRequestException(
          String.format("Insufficient balance. Available: Rs.%.2f", senderBalance));
    }

    // =========================================================================
    // FRAUD CHECK — runs before any money moves
    // =========================================================================
    FraudAssessment fraud = fraudService.assess(sender, recipient, amount, note);

    if ("BLOCK".equals(fraud.recommendation)) {
      // Surface the first human-readable signal to the user (but not internal codes)
      String reason = (fraud.signalLabels != null && !fraud.signalLabels.isEmpty())
          ? fraud.signalLabels.get(0)
          : "Transaction blocked for security reasons.";
      throw new AuthService.BadRequestException("Transaction blocked: " + reason);
    }
    // =========================================================================

    // --- Deduct and credit (atomic write) ---
    double newSenderBalance    = senderBalance - amount;
    double newRecipientBalance = recipientBalance + amount;

    for (StoredUser u : users) {
      if (Objects.equals(u.id, sender.id))    u.balance = newSenderBalance;
      if (Objects.equals(u.id, recipient.id)) u.balance = newRecipientBalance;
    }
    authService.writeUsers(users);

    // --- Record transaction with fraud metadata ---
    Transaction txn = new Transaction();
    txn.id                  = "txn_" + generateHex(8);
    txn.fromUserId          = sender.id;
    txn.fromName            = sender.name;
    txn.fromIdentifier      = sender.identifier;
    txn.toUserId            = recipient.id;
    txn.toName              = recipient.name;
    txn.toIdentifier        = recipient.identifier;
    txn.amount              = amount;
    txn.note                = note;
    txn.status              = "SUCCESS";
    txn.createdAt           = Instant.now().toString();
    txn.fraudScore          = fraud.fraudScore;
    txn.fraudRiskLevel      = fraud.riskLevel;
    txn.fraudSignals        = fraud.signals;
    txn.fraudRecommendation = fraud.recommendation;

    List<Transaction> txns = readTransactions();
    txns.add(txn);
    writeTransactions(txns);

    // --- Build response ---
    TransactionController.SendResponse resp = new TransactionController.SendResponse();
    resp.ok                  = true;
    resp.transactionId       = txn.id;
    resp.amount              = amount;
    resp.recipientName       = recipient.name;
    resp.recipientIdentifier = recipient.identifier;
    resp.newBalance          = newSenderBalance;
    resp.status              = "SUCCESS";
    resp.createdAt           = txn.createdAt;
    resp.fraudScore          = fraud.fraudScore;
    resp.fraudRiskLevel      = fraud.riskLevel;
    return resp;
  }

  /** Read all raw transactions for a specific userId — used by CreditService and FraudService. */
  public List<Transaction> getRawTransactionsForUser(String userId) {
    return readTransactions().stream()
        .filter(t -> Objects.equals(t.fromUserId, userId) || Objects.equals(t.toUserId, userId))
        .collect(Collectors.toList());
  }

  public List<TransactionController.TxnEntry> getHistory(String authHeader) {
    StoredUser       user = authService.getUserByToken(authHeader);
    List<Transaction> all = readTransactions();

    return all.stream()
        .filter(t -> Objects.equals(t.fromUserId, user.id) || Objects.equals(t.toUserId, user.id))
        .sorted((a, b) -> b.createdAt.compareTo(a.createdAt))
        .map(t -> {
          TransactionController.TxnEntry e = new TransactionController.TxnEntry();
          e.id                   = t.id;
          e.amount               = t.amount;
          e.note                 = t.note;
          e.status               = t.status;
          e.createdAt            = t.createdAt;
          e.fraudScore           = t.fraudScore;
          e.fraudRiskLevel       = t.fraudRiskLevel != null ? t.fraudRiskLevel : "LOW";
          e.fraudSignals         = t.fraudSignals;
          if (Objects.equals(t.fromUserId, user.id)) {
            e.direction                = "SENT";
            e.counterpartyName         = t.toName;
            e.counterpartyIdentifier   = t.toIdentifier;
          } else {
            e.direction                = "RECEIVED";
            e.counterpartyName         = t.fromName;
            e.counterpartyIdentifier   = t.fromIdentifier;
          }
          return e;
        })
        .collect(Collectors.toList());
  }

  // ---------------------------------------------------------------------------
  // File storage
  // ---------------------------------------------------------------------------

  private synchronized List<Transaction> readTransactions() {
    try {
      if (!Files.exists(transactionsFile)) {
        Files.write(transactionsFile, "[]".getBytes(StandardCharsets.UTF_8));
      }
      String raw = new String(Files.readAllBytes(transactionsFile), StandardCharsets.UTF_8);
      List<Transaction> list = objectMapper.readValue(raw, new TypeReference<List<Transaction>>() {});
      return list != null ? list : new ArrayList<>();
    } catch (Exception e) {
      return new ArrayList<>();
    }
  }

  private synchronized void writeTransactions(List<Transaction> txns) {
    try {
      objectMapper.writerWithDefaultPrettyPrinter().writeValue(transactionsFile.toFile(), txns);
    } catch (Exception e) {
      throw new RuntimeException("Unable to persist transactions.", e);
    }
  }

  private String generateHex(int byteCount) {
    byte[] b = new byte[byteCount];
    RANDOM.nextBytes(b);
    return AuthService.toHex(b);
  }

  // ---------------------------------------------------------------------------
  // Storage DTO (persisted to transactions.json)
  // ---------------------------------------------------------------------------

  public static class Transaction {
    public String       id;
    public String       fromUserId;
    public String       fromName;
    public String       fromIdentifier;
    public String       toUserId;
    public String       toName;
    public String       toIdentifier;
    public double       amount;
    public String       note;
    public String       status;
    public String       createdAt;
    // Fraud metadata
    public int          fraudScore;
    public String       fraudRiskLevel;
    public List<String> fraudSignals;
    public String       fraudRecommendation;
  }
}
