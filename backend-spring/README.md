# GoPay Backend (Spring Boot)

This Spring Boot service provides the auth APIs used by the current React frontend:

- `GET /health`
- `POST /api/v1/auth/signup`
- `POST /api/v1/auth/login`
- `GET /api/v1/me` (requires `Authorization: Bearer <token>`)
- `POST /api/v1/auth/logout` (requires `Authorization: Bearer <token>`)

## Run

From the `backend-spring` folder:

```bash
mvn spring-boot:run
```

The server runs on `http://localhost:8080`.

## Storage

For the demo, user/session data is file-backed and stored in:

`backend/data/users.json`
`backend/data/sessions.json`

