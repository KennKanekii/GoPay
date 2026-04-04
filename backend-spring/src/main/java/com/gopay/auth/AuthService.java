package com.gopay.auth;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Service
public class AuthService {
  private final ObjectMapper objectMapper;
  private final Path dataDir;
  private final Path usersFile;
  private final Path sessionsFile;

  private static final SecureRandom RANDOM = new SecureRandom();
  private static final char[] HEX_ARRAY = "0123456789abcdef".toCharArray();

  /** Default wallet balance given to every new account (in INR). */
  public static final double DEFAULT_BALANCE = 10_000.0;

  public AuthService(ObjectMapper objectMapper, @Value("${gopay.dataDir:./data}") String dataDir) {
    this.objectMapper = objectMapper;
    this.dataDir = Paths.get(dataDir);
    this.usersFile = this.dataDir.resolve("users.json");
    this.sessionsFile = this.dataDir.resolve("sessions.json");
  }

  /** Exposed so TransactionService can resolve its own data files in the same directory. */
  public Path getDataDir() {
    return dataDir;
  }

  public String signup(AuthController.SignupRequest body) {
    String name = Optional.ofNullable(body).map(b -> b.name).orElse("").trim();
    String identifierRaw = Optional.ofNullable(body).map(b -> b.identifier).orElse("");
    String password = Optional.ofNullable(body).map(b -> b.password).orElse("");

    if (name.length() < 2) throw new BadRequestException("Name is required.");
    if (identifierRaw.trim().length() < 3) throw new BadRequestException("Phone/email is required.");
    if (password.length() < 6) throw new BadRequestException("Password must be at least 6 characters.");

    String identifier = normalizeIdentifier(identifierRaw);
    List<StoredUser> users = readUsers();
    if (users.stream().anyMatch(u -> Objects.equals(u.identifier, identifier))) {
      throw new ConflictException("An account with that phone/email already exists.");
    }

    String salt = newHex(16);
    String userId = generateId();
    StoredUser user = new StoredUser();
    user.id = userId;
    user.name = name;
    user.identifier = identifier;
    user.passwordSalt = salt;
    user.passwordHash = hashPassword(password, salt);
    user.balance = DEFAULT_BALANCE;
    user.createdAt = Instant.now().toString();

    users.add(user);
    writeUsers(users);
    return userId;
  }

  public AuthController.LoginResponse login(AuthController.LoginRequest body) {
    String identifierRaw = Optional.ofNullable(body).map(b -> b.identifier).orElse("");
    String password = Optional.ofNullable(body).map(b -> b.password).orElse("");

    if (identifierRaw.trim().length() < 3) throw new BadRequestException("Phone/email is required.");
    if (password.length() < 6) throw new UnauthorizedException("Invalid credentials.");

    String identifier = normalizeIdentifier(identifierRaw);
    List<StoredUser> users = readUsers();
    StoredUser user = users.stream()
        .filter(u -> Objects.equals(u.identifier, identifier))
        .findFirst().orElse(null);
    if (user == null) throw new UnauthorizedException("Invalid credentials.");

    if (!Objects.equals(hashPassword(password, user.passwordSalt), user.passwordHash)) {
      throw new UnauthorizedException("Invalid credentials.");
    }

    String token = generateToken();
    List<Session> sessions = readSessions();
    Session s = new Session();
    s.token = token;
    s.userId = user.id;
    s.createdAt = Instant.now().toString();
    sessions.add(s);
    writeSessions(sessions);

    AuthController.LoginResponse resp = new AuthController.LoginResponse();
    resp.ok = true;
    resp.token = token;

    AuthController.User u = new AuthController.User();
    u.id = user.id;
    u.name = user.name;
    u.identifier = user.identifier;
    resp.user = u;

    return resp;
  }

  public AuthController.MeResponse me(String authHeader) {
    StoredUser user = getUserByToken(authHeader);

    AuthController.MeResponse resp = new AuthController.MeResponse();
    resp.id = user.id;
    resp.name = user.name;
    resp.identifier = user.identifier;
    // Seed legacy accounts (balance==0) on first /me call.
    resp.balance = user.balance > 0 ? user.balance : DEFAULT_BALANCE;
    return resp;
  }

  public void logout(String authHeader) {
    String token = extractBearerToken(authHeader);
    if (token == null) throw new UnauthorizedException("Not authenticated.");

    List<Session> sessions = readSessions();
    sessions.removeIf(s -> Objects.equals(s.token, token));
    writeSessions(sessions);
  }

  /**
   * Validates the Bearer token and returns the corresponding StoredUser.
   * Throws UnauthorizedException on any failure.
   */
  public StoredUser getUserByToken(String authHeader) {
    String token = extractBearerToken(authHeader);
    if (token == null) throw new UnauthorizedException("Not authenticated.");

    List<Session> sessions = readSessions();
    Session session = sessions.stream()
        .filter(s -> Objects.equals(s.token, token))
        .findFirst().orElse(null);
    if (session == null) throw new UnauthorizedException("Session expired.");

    List<StoredUser> users = readUsers();
    StoredUser user = users.stream()
        .filter(u -> Objects.equals(u.id, session.userId))
        .findFirst().orElse(null);
    if (user == null) throw new UnauthorizedException("Invalid user.");
    return user;
  }

  private String extractBearerToken(String authHeader) {
    if (authHeader == null) return null;
    if (authHeader.trim().isEmpty()) return null;
    String trimmed = authHeader.trim();
    if (!trimmed.regionMatches(true, 0, "Bearer ", 0, "Bearer ".length())) return null;
    String token = trimmed.substring("Bearer ".length()).trim();
    return token.isEmpty() ? null : token;
  }

  private String normalizeIdentifier(String value) {
    return value.trim().toLowerCase();
  }

  private String generateId() {
    return "user_" + newHex(8);
  }

  private String generateToken() {
    return "tok_" + newHex(16);
  }

  private String hashPassword(String password, String salt) {
    try {
      MessageDigest md = MessageDigest.getInstance("SHA-256");
      byte[] digest = md.digest((salt + ":" + password).getBytes(StandardCharsets.UTF_8));
      return toHex(digest);
    } catch (Exception e) {
      throw new RuntimeException("Unable to hash password.", e);
    }
  }

  private String newHex(int byteCount) {
    byte[] b = new byte[byteCount];
    RANDOM.nextBytes(b);
    return toHex(b);
  }

  public static String toHex(byte[] bytes) {
    char[] hexChars = new char[bytes.length * 2];
    for (int i = 0; i < bytes.length; i++) {
      int v = bytes[i] & 0xFF;
      hexChars[i * 2] = HEX_ARRAY[v >>> 4];
      hexChars[i * 2 + 1] = HEX_ARRAY[v & 0x0F];
    }
    return new String(hexChars);
  }

  private void ensureFilesExist() {
    try {
      if (!Files.exists(dataDir)) Files.createDirectories(dataDir);
      if (!Files.exists(usersFile)) Files.write(usersFile, "[]".getBytes(StandardCharsets.UTF_8));
      if (!Files.exists(sessionsFile)) Files.write(sessionsFile, "[]".getBytes(StandardCharsets.UTF_8));
    } catch (Exception e) {
      throw new RuntimeException("Unable to initialize auth storage.", e);
    }
  }

  public synchronized List<StoredUser> readUsers() {
    ensureFilesExist();
    try {
      String raw = new String(Files.readAllBytes(usersFile), StandardCharsets.UTF_8);
      List<StoredUser> users = objectMapper.readValue(raw, new TypeReference<List<StoredUser>>() {});
      return users != null ? users : new ArrayList<>();
    } catch (Exception e) {
      return new ArrayList<>();
    }
  }

  public synchronized void writeUsers(List<StoredUser> users) {
    ensureFilesExist();
    try {
      objectMapper.writerWithDefaultPrettyPrinter().writeValue(usersFile.toFile(), users);
    } catch (Exception e) {
      throw new RuntimeException("Unable to persist users.", e);
    }
  }

  private synchronized List<Session> readSessions() {
    ensureFilesExist();
    try {
      String raw = new String(Files.readAllBytes(sessionsFile), StandardCharsets.UTF_8);
      List<Session> sessions = objectMapper.readValue(raw, new TypeReference<List<Session>>() {});
      return sessions != null ? sessions : new ArrayList<>();
    } catch (Exception e) {
      return new ArrayList<>();
    }
  }

  private synchronized void writeSessions(List<Session> sessions) {
    ensureFilesExist();
    try {
      objectMapper.writerWithDefaultPrettyPrinter().writeValue(sessionsFile.toFile(), sessions);
    } catch (Exception e) {
      throw new RuntimeException("Unable to persist sessions.", e);
    }
  }

  // -------------------------------------------------------------------------
  // Storage DTOs
  // -------------------------------------------------------------------------

  public static class StoredUser {
    public String id;
    public String name;
    public String identifier;
    public String passwordSalt;
    public String passwordHash;
    public double balance;
    public String createdAt;
  }

  static class Session {
    public String token;
    public String userId;
    public String createdAt;
  }

  // -------------------------------------------------------------------------
  // Exception types (mapped to HTTP status codes by the controllers)
  // -------------------------------------------------------------------------

  public static class BadRequestException extends RuntimeException {
    public BadRequestException(String message) { super(message); }
  }

  public static class ConflictException extends RuntimeException {
    public ConflictException(String message) { super(message); }
  }

  public static class UnauthorizedException extends RuntimeException {
    public UnauthorizedException(String message) { super(message); }
  }
}
