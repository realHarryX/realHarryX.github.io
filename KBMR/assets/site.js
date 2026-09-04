(() => {
  const header = document.querySelector('[data-header]');
  const menuToggle = document.querySelector('[data-menu-toggle]');
  const menu = document.querySelector('[data-menu]');
  const navLinks = [...document.querySelectorAll('[data-nav-link]')];
  const backToTop = document.querySelector('[data-back-to-top]');
  const copyButton = document.querySelector('[data-copy-button]');
  const copyLabel = document.querySelector('[data-copy-label]');
  const copyStatus = document.querySelector('[data-copy-status]');
  const bibtex = document.querySelector('#bibtex');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  function closeMenu() {
    if (!menuToggle || !menu) return;
    menuToggle.setAttribute('aria-expanded', 'false');
    menu.classList.remove('is-open');
  }

  function setHeaderState() {
    const scrolled = window.scrollY > 12;
    header?.classList.toggle('is-scrolled', scrolled);
    backToTop?.classList.toggle('is-visible', window.scrollY > 680);
  }

  if (menuToggle && menu) {
    menuToggle.addEventListener('click', () => {
      const isOpen = menuToggle.getAttribute('aria-expanded') === 'true';
      menuToggle.setAttribute('aria-expanded', String(!isOpen));
      menu.classList.toggle('is-open', !isOpen);
    });

    menu.addEventListener('click', (event) => {
      if (event.target.closest('a')) closeMenu();
    });

    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        closeMenu();
        menuToggle.focus();
      }
    });
  }

  setHeaderState();
  window.addEventListener('scroll', setHeaderState, { passive: true });

  const sections = navLinks
    .map((link) => document.querySelector(link.getAttribute('href')))
    .filter(Boolean);

  if ('IntersectionObserver' in window && sections.length) {
    const navigationObserver = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!visible) return;
        navLinks.forEach((link) => {
          link.classList.toggle('is-active', link.getAttribute('href') === `#${visible.target.id}`);
        });
      },
      { rootMargin: '-35% 0px -55% 0px', threshold: [0.05, 0.2, 0.5] }
    );
    sections.forEach((section) => navigationObserver.observe(section));
  }

  const revealElements = [...document.querySelectorAll('.reveal')];
  if (reducedMotion.matches || !('IntersectionObserver' in window)) {
    revealElements.forEach((element) => element.classList.add('is-visible'));
  } else {
    const revealObserver = new IntersectionObserver(
      (entries, observer) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        });
      },
      { threshold: 0.12, rootMargin: '0px 0px -5% 0px' }
    );
    revealElements.forEach((element) => revealObserver.observe(element));
  }

  function selectBibtex() {
    if (!bibtex) return;
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(bibtex);
    selection.removeAllRanges();
    selection.addRange(range);
  }

  async function copyCitation() {
    if (!bibtex || !copyButton || !copyLabel || !copyStatus) return;
    const citation = bibtex.textContent.trim();
    let copied = false;

    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(citation);
        copied = true;
      }
    } catch (_) {
      copied = false;
    }

    if (!copied) {
      selectBibtex();
      try {
        copied = document.execCommand('copy');
      } catch (_) {
        copied = false;
      }
    }

    if (copied) {
      copyLabel.textContent = 'Copied';
      copyStatus.textContent = 'BibTeX copied to the clipboard.';
      copyButton.classList.add('is-copied');
      window.setTimeout(() => {
        copyLabel.textContent = 'Copy';
        copyStatus.textContent = 'Ready to copy.';
        copyButton.classList.remove('is-copied');
      }, 1800);
    } else {
      selectBibtex();
      copyLabel.textContent = 'Selected';
      copyStatus.textContent = 'Copy unavailable. BibTeX selected for manual copy.';
      window.setTimeout(() => {
        copyLabel.textContent = 'Copy';
      }, 2200);
    }
  }

  copyButton?.addEventListener('click', copyCitation);
})();
