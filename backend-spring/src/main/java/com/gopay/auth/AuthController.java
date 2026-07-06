package com.gopay.auth;

import java.util.HashMap;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RestController;

@RestController
@CrossOrigin(origins = "*")
public class AuthController {
  private final AuthService authService;

  public AuthController(AuthService authService) {
    this.authService = authService;
  }

  @GetMapping("/health")
  public Map<String, Object> health() {
    Map<String, Object> res = new HashMap<>();
    res.put("ok", true);
    res.put("service", "gopay-backend");
    return res;
  }

  @PostMapping("/api/v1/auth/signup")
  public ResponseEntity<?> signup(@RequestBody SignupRequest body) {
    try {
      String id = authService.signup(body);
      Map<String, Object> res = new HashMap<>();
      res.put("ok", true);
      res.put("id", id);
      return ResponseEntity.status(HttpStatus.CREATED).body(res);
    } catch (AuthService.BadRequestException e) {
      Map<String, Object> res = new HashMap<>();
      res.put("ok", false);
      res.put("error", e.getMessage());
      return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(res);
    } catch (AuthService.ConflictException e) {
      Map<String, Object> res = new HashMap<>();
      res.put("ok", false);
      res.put("error", e.getMessage());
      return ResponseEntity.status(HttpStatus.CONFLICT).body(res);
    }
  }

  @PostMapping("/api/v1/auth/login")
  public ResponseEntity<?> login(@RequestBody LoginRequest body) {
    try {
      LoginResponse resp = authService.login(body);
      return ResponseEntity.ok(resp);
    } catch (AuthService.BadRequestException e) {
      Map<String, Object> res = new HashMap<>();
      res.put("ok", false);
      res.put("error", e.getMessage());
      return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(res);
    } catch (AuthService.UnauthorizedException e) {
      Map<String, Object> res = new HashMap<>();
      res.put("ok", false);
      res.put("error", e.getMessage());
      return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(res);
    }
  }

  @GetMapping("/api/v1/me")
  public ResponseEntity<?> me(@RequestHeader(name = "Authorization", required = false) String authHeader) {
    try {
      MeResponse me = authService.me(authHeader);
      return ResponseEntity.ok(me);
    } catch (AuthService.UnauthorizedException e) {
      Map<String, Object> res = new HashMap<>();
      res.put("ok", false);
      res.put("error", e.getMessage());
      return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(res);
    }
  }

  @PatchMapping("/api/v1/me")
  public ResponseEntity<?> updateProfile(
      @RequestHeader(name = "Authorization", required = false) String authHeader,
      @RequestBody UpdateProfileRequest body) {
    try {
      MeResponse me = authService.updateProfile(authHeader, body);
      return ResponseEntity.ok(me);
    } catch (AuthService.UnauthorizedException e) {
      Map<String, Object> res = new HashMap<>();
      res.put("ok", false); res.put("error", e.getMessage());
      return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(res);
    } catch (AuthService.BadRequestException e) {
      Map<String, Object> res = new HashMap<>();
      res.put("ok", false); res.put("error", e.getMessage());
      return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(res);
    }
  }

  @PostMapping("/api/v1/auth/logout")
  public ResponseEntity<?> logout(@RequestHeader(name = "Authorization", required = false) String authHeader) {
    try {
      authService.logout(authHeader);
      Map<String, Object> res = new HashMap<>();
      res.put("ok", true);
      return ResponseEntity.ok(res);
    } catch (AuthService.UnauthorizedException e) {
      Map<String, Object> res = new HashMap<>();
      res.put("ok", false);
      res.put("error", e.getMessage());
      return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(res);
    }
  }

  // --- DTOs ---
  public static class SignupRequest {
    public String name;
    public String identifier;
    public String password;
    // Optional extended fields
    public String mobileNumber;
    public String vpa;
    public String bankAccount;
    public String ifscCode;
  }

  public static class LoginRequest {
    public String identifier;
    public String password;
  }

  public static class UpdateProfileRequest {
    public String name;
    public String mobileNumber;
    public String vpa;
    public String bankAccount;
    public String ifscCode;
  }

  public static class MeResponse {
    public String id;
    public String name;
    public String identifier;
    public double balance;
    public String mobileNumber;
    public String vpa;
    public String bankAccount;
    public String ifscCode;
  }

  public static class LoginResponse {
    public boolean ok;
    public String token;
    public User user;
  }

  public static class User {
    public String id;
    public String name;
    public String identifier;
  }
}

