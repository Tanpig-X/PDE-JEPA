(() => {
  const count = document.getElementById('busuanzi_value_page_pv');
  if (!count || location.hostname !== 'tanpig-x.github.io') return;

  const showUnavailable = () => {
    if (count.textContent.trim() === '—') count.textContent = 'Unavailable';
  };
  const counter = document.createElement('script');
  counter.src = 'https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js';
  counter.async = true;
  counter.onerror = showUnavailable;
  document.head.appendChild(counter);
  window.setTimeout(showUnavailable, 10000);
})();
