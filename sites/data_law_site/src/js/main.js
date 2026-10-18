document.addEventListener('DOMContentLoaded', () => {
  const toggle = document.getElementById('menu-toggle');
  const nav = document.getElementById('main-nav');

  if (toggle && nav) {
    toggle.addEventListener('click', (event) => {
      event.preventDefault();
      const isOpen = nav.classList.toggle('is-open');
      toggle.classList.toggle('active');
      document.body.style.overflow = isOpen ? 'hidden' : '';
    });
  }

  document.querySelectorAll('.expander').forEach((button) => {
    button.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();

      const group = button.closest('.nav-group');
      const panel = group?.querySelector('.dropdown-panel');
      if (!panel) return;

      const isActive = panel.classList.toggle('is-active');
      button.style.transform = isActive ? 'rotate(180deg)' : '';
    });
  });

  document.querySelectorAll('.drop-item, .nav-link').forEach((link) => {
    link.addEventListener('click', () => {
      if (window.innerWidth <= 1150 && nav && toggle) {
        nav.classList.remove('is-open');
        toggle.classList.remove('active');
        document.body.style.overflow = '';
      }
    });
  });

  const laylaPanel = document.getElementById('laylaWindow');
  const laylaInput = document.getElementById('laylaIn');
  const openLayla = (event) => {
    if (!laylaPanel) return;
    event.preventDefault();
    laylaPanel.classList.add('active');
    laylaPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    laylaInput?.focus();
  };

  document.getElementById('laylaTriggerFooter')?.addEventListener('click', openLayla);
  document.querySelectorAll('[data-trigger="layla"]').forEach((button) => {
    button.addEventListener('click', openLayla);
  });

  // Scroll animations
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('in-view');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.dl-section, .dl-card, .home-news-side__item').forEach((el) => {
    el.classList.add('animate-reveal');
    observer.observe(el);
  });
});
