// ─── Intersection Observer — AOS
const initAOS = () => {
  const els = document.querySelectorAll('[data-aos]');
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('aos-animate');
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });
  els.forEach(el => obs.observe(el));
};

// ─── Counter animation
const initCounters = () => {
  const counters = document.querySelectorAll('.stat-num[data-count]');
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      const el = e.target;
      const target = parseInt(el.dataset.count);
      let start = 0;
      const duration = 1200;
      const step = target / (duration / 16);
      const timer = setInterval(() => {
        start += step;
        if (start >= target) {
          el.textContent = target;
          clearInterval(timer);
        } else {
          el.textContent = Math.floor(start);
        }
      }, 16);
      obs.unobserve(el);
    });
  }, { threshold: 0.5 });
  counters.forEach(c => obs.observe(c));
};

// ─── Cursor glow
const initCursorGlow = () => {
  const glow = document.querySelector('.cursor-glow');
  if (!glow || window.matchMedia('(pointer: coarse)').matches) return;
  let x = 0, y = 0, tx = 0, ty = 0;
  document.addEventListener('mousemove', e => { tx = e.clientX; ty = e.clientY; });
  const animate = () => {
    x += (tx - x) * 0.12;
    y += (ty - y) * 0.12;
    glow.style.transform = `translate(${x - 250}px, ${y - 250}px)`;
    requestAnimationFrame(animate);
  };
  animate();
};

// ─── Mobile menu
const initMobileMenu = () => {
  const toggle = document.getElementById('mobileToggle');
  const nav = document.getElementById('mainNav');
  if (!toggle || !nav) return;
  toggle.addEventListener('click', () => {
    nav.classList.toggle('mobile-open');
    const icon = toggle.querySelector('i');
    icon.className = nav.classList.contains('mobile-open') ? 'fas fa-times' : 'fas fa-bars';
  });
  // Close on nav link click
  nav.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => {
      nav.classList.remove('mobile-open');
      toggle.querySelector('i').className = 'fas fa-bars';
    });
  });
};

// ─── Header scroll
const initHeader = () => {
  const header = document.getElementById('header');
  if (!header) return;
  window.addEventListener('scroll', () => {
    header.classList.toggle('scrolled', window.scrollY > 20);
  }, { passive: true });
};

// ─── Contact form
const initContactForm = () => {
  const form = document.getElementById('contactForm');
  if (!form) return;
  form.addEventListener('submit', e => {
    e.preventDefault();
    form.style.display = 'none';
    const s = document.getElementById('formSuccess');
    if (s) s.style.display = 'block';
  });
};

// ─── Init
document.addEventListener('DOMContentLoaded', () => {
  initAOS();
  initCounters();
  initCursorGlow();
  initMobileMenu();
  initHeader();
  initContactForm();
});
