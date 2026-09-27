(() => {
  'use strict';
  const root = document.getElementById('demos');
  if (!root || typeof PDE_DEMOS === 'undefined') return;
  const get = id => document.getElementById(id);
  const dataset = get('demo-dataset');
  const split = get('demo-split');
  const sample = get('demo-sample');
  const species = get('demo-species');
  const video = get('demo-video');
  const play = get('demo-play');
  const seek = get('demo-seek');
  const speed = get('demo-speed');
  const readout = get('demo-readout');
  const error = get('demo-error');
  const names = {Vorticity:'Vorticity', Wave:'Wave-2D', GS:'Gray–Scott', HeterNS:'HeterNS', Advection:'Advection', Burgers:'Burgers', Heat:'Heat', 'Wave-B':'Wave-B', Combined:'Combined'};
  const splits = {ID:'In-distribution (ID)', OOD:'Out-of-distribution (OOD)', OOD_force:'OOD · forcing', OOD_nu_interpolation:'OOD · viscosity interpolation', OOD_nu_extrapolation:'OOD · viscosity extrapolation'};
  let current;
  let requested = false;
  let desiredFrame = null;
  let hasFrame = false;

  function setOptions(select, values, label, preferred) {
    select.replaceChildren(...values.map(value => new Option(label(value), value)));
    select.value = values.includes(preferred) ? preferred : values[0];
  }
  const unique = values => [...new Set(values)];

  function updateReadout() {
    const frame = Math.min(current.frames - 1, Math.floor((video.currentTime || 0) * current.fps));
    seek.value = String(frame);
    seek.setAttribute('aria-valuetext', `Frame ${frame + 1} of ${current.frames}`);
    readout.textContent = hasFrame ? `Frame ${String(frame + 1).padStart(2, '0')} / ${current.frames}` : `Preview · ${current.frames} frames`;
    play.textContent = video.paused ? '▶ Play' : 'Ⅱ Pause';
    play.setAttribute('aria-label', video.paused ? 'Play demo' : 'Pause demo');
  }

  function ensureLoaded() {
    if (!requested) {
      requested = true;
      video.load();
    }
  }

  function render() {
    const matches = PDE_DEMOS.filter(item => item.dataset === dataset.value && item.split === split.value);
    current = matches.find(item => item.sample === sample.value && (item.channel === null || item.channel === species.value));
    if (!current) return;
    video.pause();
    requested = false;
    desiredFrame = null;
    hasFrame = false;
    video.preload = 'none';
    video.poster = current.poster;
    video.src = current.src;
    video.setAttribute('aria-label', current.label);
    seek.max = String(current.frames - 1);
    seek.value = '0';
    error.hidden = true;
    get('demo-title').textContent = names[current.dataset] + (current.channel ? ` · species ${current.channel}` : '');
    get('demo-parameters').textContent = current.parameters;
    get('demo-condition').textContent = splits[current.split];
    get('demo-method').textContent = 'Model output';
    get('demo-sample-label').textContent = `Sample ${current.sample} · ${current.frames} frames · ${current.frames / current.fps} s`;
    get('demo-count').textContent = `${matches.length} videos in this setting`;
    get('demo-download').href = current.src;
    get('demo-download').download = `${current.dataset}-${current.src.split('/').pop()}`;
    get('demo-open').href = current.src;
    updateReadout();
  }

  function setSamples(preferred = sample.value) {
    const matches = PDE_DEMOS.filter(item => item.dataset === dataset.value && item.split === split.value);
    setOptions(sample, unique(matches.map(item => item.sample)), value => `Sample ${value}`, preferred);
    get('demo-species-field').hidden = dataset.value !== 'GS';
    render();
  }

  function setDataset(preferredSplit = split.value) {
    const values = unique(PDE_DEMOS.filter(item => item.dataset === dataset.value).map(item => item.split));
    setOptions(split, values, value => splits[value], preferredSplit);
    setSamples();
  }

  setOptions(dataset, Object.keys(names), value => names[value], 'Vorticity');
  setDataset('OOD');
  dataset.addEventListener('change', () => setDataset());
  split.addEventListener('change', () => setSamples());
  sample.addEventListener('change', render);
  species.addEventListener('change', render);

  play.addEventListener('click', async () => {
    if (!video.paused) { video.pause(); return; }
    const key = current.key;
    ensureLoaded();
    try {
      await video.play();
    } catch (reason) {
      if (key !== current.key || reason.name === 'AbortError') return;
      error.textContent = 'Playback could not start. Try Play again, or open the MP4 below.';
      error.hidden = false;
    }
  });
  seek.addEventListener('input', () => {
    video.pause();
    desiredFrame = Number(seek.value);
    ensureLoaded();
    if (video.readyState >= 1) {
      video.currentTime = (desiredFrame + 0.01) / current.fps;
      desiredFrame = null;
    }
  });
  speed.addEventListener('change', () => { video.playbackRate = Number(speed.value); });
  video.addEventListener('loadedmetadata', () => {
    video.playbackRate = Number(speed.value);
    if (desiredFrame !== null) {
      video.currentTime = (desiredFrame + 0.01) / current.fps;
      desiredFrame = null;
    }
  });
  video.addEventListener('loadeddata', () => { hasFrame = true; updateReadout(); });
  ['play', 'pause', 'timeupdate', 'seeked', 'ended'].forEach(event => video.addEventListener(event, updateReadout));
  video.addEventListener('error', () => {
    error.textContent = 'This video could not load. Open or download the original MP4 below.';
    error.hidden = false;
  });
  get('demo-fullscreen').addEventListener('click', async () => {
    const screen = get('demo-screen');
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if (screen.requestFullscreen) await screen.requestFullscreen();
      else if (video.webkitEnterFullscreen) video.webkitEnterFullscreen();
      else {
        error.textContent = 'Full screen is unavailable here. Use Open MP4 for a larger view.';
        error.hidden = false;
      }
    } catch {
      error.textContent = 'Full screen is unavailable here. Use Open MP4 for a larger view.';
      error.hidden = false;
    }
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) video.pause(); });
  new IntersectionObserver(entries => {
    if (!entries[0].isIntersecting) video.pause();
  }, {threshold: 0}).observe(video);
  video.controls = false;
  get('demo-controls').hidden = false;
  get('demo-filters').hidden = false;
})();
