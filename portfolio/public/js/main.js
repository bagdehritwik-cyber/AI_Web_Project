const form = document.getElementById('contactForm');
const alertBox = document.getElementById('formAlert');
const yearElement = document.getElementById('year');

const isValidEmail = (email) => {
  if (email.length > 254) return false;
  const atIndex = email.indexOf('@');
  if (atIndex <= 0 || atIndex !== email.lastIndexOf('@') || atIndex >= email.length - 1) return false;
  const domain = email.slice(atIndex + 1);
  if (domain.startsWith('.') || domain.endsWith('.') || domain.includes('..')) return false;
  const dotIndex = domain.lastIndexOf('.');
  return dotIndex > 0 && dotIndex < domain.length - 1;
};

const showAlert = (message, type) => {
  alertBox.className = `alert alert-${type}`;
  alertBox.textContent = message;
  alertBox.classList.remove('d-none');
};

yearElement.textContent = new Date().getFullYear();

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  const payload = {
    name: String(formData.get('name') || '').trim(),
    email: String(formData.get('email') || '').trim(),
    message: String(formData.get('message') || '').trim(),
  };

  if (!payload.name || !payload.email || !payload.message) {
    showAlert('Please complete all fields.', 'warning');
    return;
  }

  if (!isValidEmail(payload.email)) {
    showAlert('Please enter a valid email address.', 'warning');
    return;
  }

  try {
    const response = await fetch('/api/contact', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const result = await response.json();

    if (!response.ok || !result.success) {
      showAlert(result.message || 'Could not submit inquiry.', 'danger');
      return;
    }

    form.reset();
    showAlert('Inquiry submitted successfully. Thank you!', 'success');
  } catch (error) {
    showAlert('Network error. Please try again.', 'danger');
  }
});
