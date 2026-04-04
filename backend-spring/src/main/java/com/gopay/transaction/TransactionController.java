package com.gopay.transaction;

import com.gopay.auth.AuthService;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RestController;

@RestController
@CrossOrigin(origins = "*")
public class TransactionController {

  private final TransactionService transactionService;

  public TransactionController(TransactionService transactionService) {
    this.transactionService = transactionService;
  }

  /** GET /api/v1/wallet/balance */
  @GetMapping("/api/v1/wallet/balance")
  public ResponseEntity<?> getBalance(
      @RequestHeader(name = "Authorization", required = false) String authHeader) {
    try {
      return ResponseEntity.ok(transactionService.getBalance(authHeader));
    } catch (AuthService.UnauthorizedException e) {
      return err(HttpStatus.UNAUTHORIZED, e.getMessage());
    }
  }

  /** POST /api/v1/transactions/send */
  @PostMapping("/api/v1/transactions/send")
  public ResponseEntity<?> send(
      @RequestHeader(name = "Authorization", required = false) String authHeader,
      @RequestBody SendRequest body) {
    try {
      return ResponseEntity.ok(transactionService.send(authHeader, body));
    } catch (AuthService.UnauthorizedException e) {
      return err(HttpStatus.UNAUTHORIZED, e.getMessage());
    } catch (AuthService.BadRequestException e) {
      return err(HttpStatus.BAD_REQUEST, e.getMessage());
    }
  }

  /** GET /api/v1/transactions */
  @GetMapping("/api/v1/transactions")
  public ResponseEntity<?> getHistory(
      @RequestHeader(name = "Authorization", required = false) String authHeader) {
    try {
      List<TxnEntry> history = transactionService.getHistory(authHeader);
      return ResponseEntity.ok(history);
    } catch (AuthService.UnauthorizedException e) {
      return err(HttpStatus.UNAUTHORIZED, e.getMessage());
    }
  }

  private ResponseEntity<Map<String, Object>> err(HttpStatus status, String message) {
    Map<String, Object> body = new HashMap<>();
    body.put("ok", false);
    body.put("error", message);
    return ResponseEntity.status(status).body(body);
  }

  // ---------------------------------------------------------------------------
  // Request / Response DTOs
  // ---------------------------------------------------------------------------

  public static class SendRequest {
    public String recipientIdentifier;
    public double amount;
    public String note;
  }

  public static class SendResponse {
    public boolean ok;
    public String transactionId;
    public double amount;
    public String recipientName;
    public String recipientIdentifier;
    public double newBalance;
    public String status;
    public String createdAt;
  }

  public static class BalanceResponse {
    public double balance;
    public String currency;
  }

  public static class TxnEntry {
    public String id;
    /** "SENT" or "RECEIVED" */
    public String direction;
    public double amount;
    public String counterpartyName;
    public String counterpartyIdentifier;
    public String note;
    public String status;
    public String createdAt;
  }
}
