const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const btnAnalyze = document.getElementById('btnAnalyze');
const btnCamera = document.getElementById('btnCamera');
const cameraWrap = document.getElementById('camera');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const btnCapture = document.getElementById('btnCapture');
const btnCloseCam = document.getElementById('btnCloseCam');

const results = document.getElementById('results');
const placeholder = document.getElementById('placeholder');
const preview = document.getElementById('preview');
const predEl = document.getElementById('pred');
const confEl = document.getElementById('conf');
const topkEl = document.getElementById('topk');
const diagEl = document.getElementById('diag');
const locEl = document.getElementById('loc');
// No address element now; only show lat/lon

let currentBlob = null;
let currentLocation = null;
let currentAddress = null; // unused now

function showResults(){
  placeholder.hidden = true;
  results.hidden = false;
}

function setPreviewFromBlob(blob){
  currentBlob = blob;
  const url = URL.createObjectURL(blob);
  preview.src = url;
  showResults();
  // Auto request location when an image is selected/captured
  autoGetLocation();
}

// Drag & drop
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('hover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('hover'));
dropzone.addEventListener('drop', e => {
  e.preventDefault(); dropzone.classList.remove('hover');
  const file = e.dataTransfer.files[0];
  if(file) setPreviewFromBlob(file);
});
fileInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if(file) setPreviewFromBlob(file);
});

// Camera
let mediaStream;
btnCamera.addEventListener('click', async () => {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    video.srcObject = mediaStream;
    cameraWrap.hidden = false;
  } catch (err) {
    alert('Camera access denied or unavailable: ' + err.message);
  }
});
btnCapture.addEventListener('click', () => {
  const w = video.videoWidth, h = video.videoHeight;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, w, h);
  canvas.toBlob(blob => {
    setPreviewFromBlob(blob);
  }, 'image/jpeg', 0.9);
});
btnCloseCam.addEventListener('click', () => {
  cameraWrap.hidden = true;
  if(mediaStream){ mediaStream.getTracks().forEach(t => t.stop()); }
});

// Auto geolocation on image selection/capture
async function autoGetLocation(){
  if(!navigator.geolocation){
    return; // silently skip
  }
  navigator.geolocation.getCurrentPosition(pos => {
    currentLocation = { lat: pos.coords.latitude, lon: pos.coords.longitude, accuracy: pos.coords.accuracy };
    locEl.textContent = `Lat ${currentLocation.lat.toFixed(5)}, Lon ${currentLocation.lon.toFixed(5)}`;
  }, _err => {
    // do nothing if user denies
  });
}

// Analysis
btnAnalyze.addEventListener('click', async () => {
  if(!currentBlob){
    alert('Please upload or capture a plant image first.');
    return;
  }
  const fd = new FormData();
  fd.append('image', currentBlob, 'plant.jpg');
  if(currentLocation) fd.append('location', JSON.stringify(currentLocation));
  // Don't append address anymore; only lat/lon

  predEl.textContent = 'Analyzing…';
  confEl.textContent = '';
  diagEl.textContent = '';
  topkEl.innerHTML = '';
  showResults();

  try{
    const r = await fetch('/analyze', { method: 'POST', body: fd });
    const j = await r.json();
    if(!r.ok){
      alert(j.error || j.message || 'Analysis failed');
      return;
    }
    if(j.message && j.message.includes('Model not available')){
      predEl.textContent = 'Model not available yet';
      diagEl.textContent = 'Run training first (see README).';
      return;
    }
    predEl.textContent = `Prediction: ${j.prediction}`;
    confEl.textContent = `Confidence: ${(j.confidence*100).toFixed(2)}%`;
    diagEl.textContent = j.diagnosis || '';
  topkEl.innerHTML = (j.topk||[]).map(t => `<li>${t.label} – ${(t.confidence*100).toFixed(1)}%</li>`).join('');
    if(j.receivedLocation){
      const loc = j.receivedLocation;
      locEl.textContent = `Lat ${Number(loc.lat).toFixed(5)}, Lon ${Number(loc.lon).toFixed(5)}`;
    }
  }catch(e){
    alert('Network error: ' + e.message);
  }
});
