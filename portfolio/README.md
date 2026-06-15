# Portfolio Module

Professional portfolio website module for **Hritwik Bagde**.

## Features

- Responsive landing page (Bootstrap)
- About/Bio section
- Featured project embeds (YouTube + Instagram)
- Contact form with frontend + backend email validation
- Contact submission email notifications to `hritwikbagde4@gmail.com`
- SQLite storage for all inquiries
- Admin dashboard to view all contact requests
- Excel export (`.xlsx`) with name, email, message, date, time

## Setup

```bash
cd portfolio
cp .env.example .env
npm install
npm start
```

Server runs at `http://localhost:3000` by default.

Set a secure `ADMIN_TOKEN` in `.env` and open the dashboard at:

`http://localhost:3000/admin?token=YOUR_ADMIN_TOKEN`

## Routes

- `GET /` - Portfolio website
- `POST /api/contact` - Submit inquiry
- `GET /admin` - Admin dashboard
- `GET /api/admin/contacts` - Contact requests JSON
- `GET /api/admin/export` - Download Excel export
