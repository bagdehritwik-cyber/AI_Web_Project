const express = require('express');

const router = express.Router();

const isValidEmail = (email) => {
  if (email.length > 254) return false;

  const atIndex = email.indexOf('@');
  if (atIndex <= 0 || atIndex !== email.lastIndexOf('@') || atIndex >= email.length - 1) {
    return false;
  }

  const domain = email.slice(atIndex + 1);
  if (domain.startsWith('.') || domain.endsWith('.') || domain.includes('..')) {
    return false;
  }

  const dotIndex = domain.lastIndexOf('.');
  return dotIndex > 0 && dotIndex < domain.length - 1;
};

router.post('/', async (req, res) => {
  const name = (req.body.name || '').trim();
  const email = (req.body.email || '').trim();
  const message = (req.body.message || '').trim();

  if (!name || !email || !message) {
    return res.status(400).json({ success: false, message: 'Name, email, and message are required.' });
  }

  if (!isValidEmail(email)) {
    return res.status(400).json({ success: false, message: 'Please provide a valid email address.' });
  }

  try {
    await req.app.locals.db.insertContact(name, email, message);
    await req.app.locals.sendContactEmail({ name, email, message });

    return res.json({ success: true, message: 'Thank you! Your inquiry has been submitted.' });
  } catch (error) {
    console.error('Contact submission failed:', error?.message || 'Unknown error');
    return res.status(500).json({ success: false, message: 'Failed to submit inquiry. Please try again later.' });
  }
});

module.exports = router;
