(() => {
  'use strict';
  const slides = [...document.querySelectorAll('.slide')];
  const counter = document.querySelector('#counter');
  const progress = document.querySelector('#progressBar');
  const notesPanel = document.querySelector('#notesPanel');
  let index = Math.max(0, Math.min(slides.length - 1, Number(location.hash.replace('#', '')) - 1 || 0));

  function show(next, updateHash = true) {
    index = (next + slides.length) % slides.length;
    slides.forEach((slide, i) => slide.classList.toggle('active', i === index));
    counter.textContent = `${String(index + 1).padStart(2, '0')} / ${String(slides.length).padStart(2, '0')}`;
    progress.style.width = `${(index + 1) / slides.length * 100}%`;
    notesPanel.querySelector('p').textContent = slides[index].querySelector('.notes')?.textContent || 'No speaker notes for this slide.';
    document.title = `${slides[index].dataset.title} — AI Race`;
    if (updateHash) history.replaceState(null, '', `#${index + 1}`);
  }

  function toggleOverview() {
    const overview = document.body.classList.toggle('overview');
    document.querySelector('#overviewBtn').classList.toggle('active', overview);
    slides.forEach((slide, i) => slide.onclick = overview ? () => { document.body.classList.remove('overview'); show(i); } : null);
  }

  function toggleNotes() {
    notesPanel.classList.toggle('open');
    document.querySelector('#notesBtn').classList.toggle('active', notesPanel.classList.contains('open'));
  }

  document.querySelector('#prevBtn').addEventListener('click', () => show(index - 1));
  document.querySelector('#nextBtn').addEventListener('click', () => show(index + 1));
  document.querySelector('#overviewBtn').addEventListener('click', toggleOverview);
  document.querySelector('#notesBtn').addEventListener('click', toggleNotes);
  document.querySelector('#fullscreenBtn').addEventListener('click', () => document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen());
  window.addEventListener('hashchange', () => show(Number(location.hash.replace('#', '')) - 1, false));
  window.addEventListener('keydown', event => {
    if (['ArrowRight', 'ArrowDown', 'PageDown', ' '].includes(event.key)) { event.preventDefault(); show(index + 1); }
    if (['ArrowLeft', 'ArrowUp', 'PageUp'].includes(event.key)) { event.preventDefault(); show(index - 1); }
    if (event.key === 'Home') show(0);
    if (event.key === 'End') show(slides.length - 1);
    if (event.key.toLowerCase() === 'o') toggleOverview();
    if (event.key.toLowerCase() === 'n') toggleNotes();
    if (event.key.toLowerCase() === 'f') document.querySelector('#fullscreenBtn').click();
    if (event.key === 'Escape' && document.body.classList.contains('overview')) toggleOverview();
  });
  show(index, false);
})();
