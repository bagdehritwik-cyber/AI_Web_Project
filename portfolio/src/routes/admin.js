const express = require('express');
const path = require('path');
const ExcelJS = require('exceljs');
const { rateLimit } = require('express-rate-limit');

const router = express.Router();

const escapeExcelFormula = (value) => {
  if (typeof value !== 'string') return value;
  return /^[=+\-@]/.test(value) ? `'${value}` : value;
};

router.use((req, res, next) => {
  const headerToken = req.get('x-admin-token');
  const authHeader = req.get('authorization') || '';
  const bearerToken = authHeader.startsWith('Bearer ') ? authHeader.slice(7).trim() : '';
  const queryToken = typeof req.query.token === 'string' ? req.query.token : '';
  const providedToken = headerToken || bearerToken || queryToken;

  if (!providedToken || providedToken !== req.app.locals.adminToken) {
    return res.status(401).json({ success: false, message: 'Unauthorized admin access.' });
  }

  return next();
});

router.use(rateLimit({
  windowMs: 60 * 1000,
  limit: 30,
  standardHeaders: true,
  legacyHeaders: false,
  validate: false,
  message: { success: false, message: 'Too many requests. Please try again later.' },
}));

router.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '..', '..', 'public', 'admin.html'));
});

router.get('/contacts', async (req, res) => {
  try {
    const contacts = await req.app.locals.db.getContacts();
    return res.json({ success: true, data: contacts });
  } catch (error) {
    console.error('Failed fetching contacts:', error);
    return res.status(500).json({ success: false, message: 'Failed to fetch contact requests.' });
  }
});

router.get('/export', async (req, res) => {
  try {
    const contacts = await req.app.locals.db.getContacts();
    const workbook = new ExcelJS.Workbook();
    const sheet = workbook.addWorksheet('Contact Requests');

    sheet.columns = [
      { header: 'Name', key: 'name', width: 25 },
      { header: 'Email', key: 'email', width: 35 },
      { header: 'Message', key: 'message', width: 50 },
      { header: 'Date', key: 'date', width: 15 },
      { header: 'Time', key: 'time', width: 12 },
    ];

    contacts.forEach((contact) => {
      const createdAt = new Date(contact.created_at);
      const isValidDate = !Number.isNaN(createdAt.getTime());
      const date = isValidDate ? createdAt.toISOString().slice(0, 10) : contact.created_at;
      const time = isValidDate ? createdAt.toTimeString().slice(0, 8) : '';

      sheet.addRow({
        name: escapeExcelFormula(contact.name),
        email: escapeExcelFormula(contact.email),
        message: escapeExcelFormula(contact.message),
        date,
        time,
      });
    });

    res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
    res.setHeader('Content-Disposition', 'attachment; filename="contact-requests.xlsx"');

    await workbook.xlsx.write(res);
    res.end();
  } catch (error) {
    console.error('Failed exporting contacts:', error);
    return res.status(500).json({ success: false, message: 'Failed to export contact requests.' });
  }
});

module.exports = router;
