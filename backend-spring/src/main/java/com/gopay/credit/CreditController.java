package com.gopay.credit;

import com.gopay.auth.AuthService;
import java.util.HashMap;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RestController;

@RestController
@CrossOrigin(origins = "*")
public class CreditController {

  private final CreditService creditService;

  public CreditController(CreditService creditService) {
    this.creditService = creditService;
  }

  /**
   * GET /api/v1/credit/score
   * Returns a credit score + breakdown for the authenticated user.
   * Calls the Python ML service; falls back to rule-based scoring if unavailable.
   */
  @GetMapping("/api/v1/credit/score")
  public ResponseEntity<?> score(
      @RequestHeader(name = "Authorization", required = false) String authHeader) {
    try {
      CreditScoreResponse resp = creditService.getScore(authHeader);
      return ResponseEntity.ok(resp);
    } catch (AuthService.UnauthorizedException e) {
      return err(HttpStatus.UNAUTHORIZED, e.getMessage());
    } catch (Exception e) {
      return err(HttpStatus.INTERNAL_SERVER_ERROR, "Could not compute credit score.");
    }
  }

  // ---------------------------------------------------------------------------
  // DTOs
  // ---------------------------------------------------------------------------

  public static class CreditScoreResponse {
    public String userId;
    public String name;
    public int    score;
    public String riskBand;   // EXCELLENT / VERY_GOOD / GOOD / FAIR / POOR
    public String colour;     // hex colour for the gauge
    public String tip;        // human-readable advice
    public String model;      // "ml_gradient_boosting" or "rule_based_fallback"
    public Map<String, Object> breakdown;  // per-factor 0-100 scores
    public Map<String, Object> features;  // raw feature values shown on the report
  }

  // ---------------------------------------------------------------------------
  // Error helper
  // ---------------------------------------------------------------------------

  private ResponseEntity<Map<String, Object>> err(HttpStatus status, String message) {
    Map<String, Object> body = new HashMap<>();
    body.put("ok", false);
    body.put("error", message);
    return ResponseEntity.status(status).body(body);
  }
}
