(() => {
  const badge = document.getElementById('page-views-badge');
  const status = document.getElementById('page-views-status');
  const retry = document.getElementById('page-views-retry');
  if (!badge || !status || !retry || location.hostname !== 'tanpig-x.github.io') return;

  let loading = false;
  let timeout;

  const showFailure = () => {
    window.clearTimeout(timeout);
    loading = false;
    badge.hidden = true;
    status.hidden = false;
    status.textContent = 'Page views · Temporarily offline';
    retry.hidden = false;
  };

  badge.onload = () => {
    if (!badge.naturalWidth) return showFailure();
    window.clearTimeout(timeout);
    loading = false;
    badge.hidden = false;
    status.hidden = true;
    retry.hidden = true;
  };
  badge.onerror = showFailure;

  const load = () => {
    if (loading) return;
    loading = true;
    badge.hidden = true;
    status.hidden = false;
    status.textContent = 'Page views · Loading…';
    retry.hidden = true;
    // One badge request records one view. Do not poll or retry automatically.
    timeout = window.setTimeout(showFailure, 15000);
    badge.src = `${badge.dataset.src}&_=${Date.now()}`;
  };

  retry.addEventListener('click', load);
  load();
})();
