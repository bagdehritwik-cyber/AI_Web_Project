const path = require('path');
const express = require('express');
const nodemailer = require('nodemailer');
const dotenv = require('dotenv');

dotenv.config();

const db = require('./db');
const contactRoutes = require('./routes/contact');
const adminRoutes = require('./routes/admin');

const app = express();
const contactReceiver = process.env.CONTACT_RECEIVER;
const adminToken = process.env.ADMIN_TOKEN;

if (!contactReceiver) {
  throw new Error('Missing CONTACT_RECEIVER environment variable.');
}

if (!adminToken) {
  throw new Error('Missing ADMIN_TOKEN environment variable.');
}

const createEmailTransport = () => {
  const { SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_SECURE } = process.env;

  if (SMTP_HOST && SMTP_PORT && SMTP_USER && SMTP_PASS) {
    return nodemailer.createTransport({
      host: SMTP_HOST,
      port: Number(SMTP_PORT),
      secure: SMTP_SECURE === 'true',
      auth: { user: SMTP_USER, pass: SMTP_PASS },
    });
  }

  return nodemailer.createTransport({ jsonTransport: true });
};

const mailer = createEmailTransport();

app.locals.db = db;
app.locals.adminToken = adminToken;
app.locals.sendContactEmail = async ({ name, email, message }) => {
  await mailer.sendMail({
    from: process.env.SMTP_FROM || process.env.SMTP_USER || 'no-reply@portfolio.local',
    to: contactReceiver,
    subject: `New Portfolio Inquiry from ${name}`,
    text: `Name: ${name}\nEmail: ${email}\n\nMessage:\n${message}`,
  });
};

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, '..', 'public')));

app.use('/api/contact', contactRoutes);
app.use('/admin', adminRoutes);
app.use('/api/admin', adminRoutes);

app.get('/health', (req, res) => {
  res.json({ ok: true });
});

const PORT = Number(process.env.PORT) || 3000;
app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`Portfolio server running on http://localhost:${PORT}`);
});
