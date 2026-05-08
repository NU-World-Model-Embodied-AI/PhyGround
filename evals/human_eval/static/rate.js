function confirmSkip(form) {
  var reason = form.querySelector('.skip-reason-input').value.trim();
  if (!reason) {
    alert('Please enter a skip reason.');
    return false;
  }
  return confirm('Are you sure you want to skip this video?\n\nReason: ' + reason);
}

var _scoreColors = {1: '#ef4444', 2: '#f97316', 3: '#eab308', 4: '#16a34a', 5: '#15803d'};

function _applyScoreColor(btn) {
  var v = parseInt(btn.dataset.value);
  btn.style.background = _scoreColors[v] || '#888';
  btn.style.borderColor = 'transparent';
  btn.style.color = '#fff';
}

function _clearScoreColor(btn) {
  btn.style.background = '';
  btn.style.borderColor = '';
  btn.style.color = '';
}

function selectScore(btn) {
  const row = btn.closest('.dim-row');
  row.querySelectorAll('.score-btn').forEach(b => { b.classList.remove('selected'); _clearScoreColor(b); });
  btn.classList.add('selected');
  _applyScoreColor(btn);
  row.querySelector('input[type="hidden"]').value = btn.dataset.value;
  checkFormComplete();
}

// Color pre-selected buttons on page load
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.score-btn.selected').forEach(_applyScoreColor);
});

function _updateSubmitBtn(form, selector) {
  var allFilled = Array.from(form.querySelectorAll('.score-input')).every(function(i) {
    if (i.closest('.physical-dim-row')) return true;  // physical dims are optional
    return i.value;
  });
  var btn = form.querySelector(selector);
  if (btn) btn.disabled = !allFilled;
}

function checkFormComplete() {
  // Save button: always enabled (allow partial saves midway)
  // Next button: always enabled (users can skip ahead)
}

// --- Play count & stay time tracking ---

var _playCountMap = {};   // vid -> count
var _stayStartMap = {};   // vid -> timestamp (ms)

function _incPlayCount(vid, card) {
  if (!vid || !card) return;
  _playCountMap[vid] = (_playCountMap[vid] || 0) + 1;
  var inp = card.querySelector('.play-count-input');
  if (inp) inp.value = _playCountMap[vid];
}

function _updateStaySeconds(vid, card) {
  if (!vid || !_stayStartMap[vid] || !card) return;
  var elapsed = (Date.now() - _stayStartMap[vid]) / 1000;
  var inp = card.querySelector('.stay-seconds-input');
  if (inp) inp.value = elapsed.toFixed(1);
}

// --- Custom video controls (comparison mode: hide duration) ---

var _progressRAF = null;
var _videoWraps = [];

function _startProgressLoop() {
  if (_progressRAF) return;
  function tick() {
    _videoWraps.forEach(function(wrap) {
      var video = wrap.querySelector('video');
      var bar = wrap.querySelector('.vc-progress');
      if (video && bar && video.duration) {
        bar.style.width = (video.currentTime / video.duration * 100) + '%';
      }
    });
    var anyPlaying = _videoWraps.some(function(wrap) {
      var v = wrap.querySelector('video');
      return v && !v.paused && !v.ended;
    });
    if (anyPlaying) {
      _progressRAF = requestAnimationFrame(tick);
    } else {
      _progressRAF = null;
    }
  }
  _progressRAF = requestAnimationFrame(tick);
}

function togglePlay(btn) {
  var video = btn.closest('.compare-video-wrap').querySelector('video');
  if (video.paused) {
    var card = btn.closest('.compare-card');
    _incPlayCount(card ? card.dataset.vid : null, card);
    video.play();
    btn.textContent = '\u23F8';
    _startProgressLoop();
  } else {
    video.pause();
    btn.textContent = '\u25B6';
  }
}

function toggleMute(btn) {
  var video = btn.closest('.compare-video-wrap').querySelector('video');
  video.muted = !video.muted;
  btn.textContent = video.muted ? '\uD83D\uDD07' : '\uD83D\uDD09';
}

function seekVideo(e, bar) {
  var video = bar.closest('.compare-video-wrap').querySelector('video');
  var rect = bar.getBoundingClientRect();
  var pct = (e.clientX - rect.left) / rect.width;
  video.currentTime = pct * video.duration;
  var progressBar = bar.querySelector('.vc-progress');
  if (progressBar) progressBar.style.width = (pct * 100) + '%';
}

document.addEventListener('DOMContentLoaded', function() {
  checkFormComplete();

  // Cache video wraps for rAF progress loop
  _videoWraps = Array.from(document.querySelectorAll('.compare-video-wrap'));

  // Reset play button when video ends & click-to-play on video itself
  _videoWraps.forEach(function(wrap) {
    var video = wrap.querySelector('video');
    if (video) {
      video.addEventListener('ended', function() {
        var btn = wrap.querySelector('.vc-play');
        if (btn) btn.textContent = '\u25B6';
      });
      video.style.cursor = 'pointer';
      video.addEventListener('click', function() {
        var btn = wrap.querySelector('.vc-play');
        if (btn) togglePlay(btn);
      });
    }
  });

  // Start stay timer for each card
  var now = Date.now();
  document.querySelectorAll('.compare-card').forEach(function(card) {
    var vid = card.dataset.vid;
    if (vid) _stayStartMap[vid] = now;
  });

});

function _showToast(msg) {
  var toast = document.getElementById('save-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'save-toast';
    toast.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:rgba(0,0,0,0.8);color:#fff;padding:16px 32px;border-radius:8px;font-size:1.2rem;z-index:9999;pointer-events:none;opacity:0;transition:opacity 0.3s;';
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.style.opacity = '1';
  setTimeout(function() { toast.style.opacity = '0'; }, 1500);
}

function _collectFormData(action) {
  var form = document.getElementById('submit-all-form');
  if (!form) return null;
  // Flush stay seconds
  document.querySelectorAll('.compare-card').forEach(function(card) {
    var vid = card.dataset.vid;
    if (vid) _updateStaySeconds(vid, card);
  });
  var data = new FormData(form);
  data.set('action', action);
  return { url: form.action, data: data };
}

function doSave() {
  var fd = _collectFormData('save');
  if (!fd) return;
  fetch(fd.url, { method: 'POST', body: fd.data })
    .then(function(r) { if (r.ok) _showToast('Saved!'); else _showToast('Error'); })
    .catch(function() { _showToast('Error'); });
}

function doNext() {
  var fd = _collectFormData('next');
  if (!fd) {
    // Fallback: go to task list if form not found
    window.location.href = '/tasks';
    return;
  }
  fetch(fd.url, { method: 'POST', body: fd.data })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.redirect) window.location.href = data.redirect;
      else window.location.href = '/tasks';
    })
    .catch(function() {
      // On any error, still navigate to next
      window.location.href = '/tasks';
    });
}
