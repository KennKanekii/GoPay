package com.gopay.fraud;

import com.gopay.auth.AuthService;
import com.gopay.auth.AuthService.StoredUser;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RestController;

@RestController
@CrossOrigin(origins = "*")
public class FraudController {

  private final FraudService fraudService;
  private final AuthService  authService;

  public FraudController(FraudService fraudService, AuthService authService) {
    this.fraudService = fraudService;
    this.authService  = authService;
  }

  /**
   * GET /api/v1/fraud/velocity
   * Returns the authenticated user's current velocity metrics vs limits.
   * Useful for the fraud dashboard — shows how close to limits the user is.
   */
  @GetMapping("/api/v1/fraud/velocity")
  public ResponseEntity<?> velocity(
      @RequestHeader(name = "Authorization", required = false) String authHeader) {
    try {
      StoredUser user = authService.getUserByToken(authHeader);
      return ResponseEntity.ok(fraudService.getVelocitySummary(user));
    } catch (AuthService.UnauthorizedException e) {
      return err(HttpStatus.UNAUTHORIZED, e.getMessage());
    }
  }

  /**
   * GET /api/v1/fraud/events
   * Returns the fraud assessment log for the authenticated user.
   * Includes all transactions that were scored — ALLOW, REVIEW, and BLOCK.
   */
  @GetMapping("/api/v1/fraud/events")
  public ResponseEntity<?> events(
      @RequestHeader(name = "Authorization", required = false) String authHeader) {
    try {
      StoredUser user = authService.getUserByToken(authHeader);
      List<FraudService.FraudEvent> events = fraudService.getEventsForUser(user.id);
      return ResponseEntity.ok(events);
    } catch (AuthService.UnauthorizedException e) {
      return err(HttpStatus.UNAUTHORIZED, e.getMessage());
    }
  }

  // -------------------------------------------------------------------------
  // Error helper
  // -------------------------------------------------------------------------
  private ResponseEntity<Map<String, Object>> err(HttpStatus status, String message) {
    Map<String, Object> body = new HashMap<>();
    body.put("ok", false);
    body.put("error", message);
    return ResponseEntity.status(status).body(body);
  }
}
