
let status = document.getElementById('status');
status.innerHTML = "Loading models...";
let imageUpload = document.getElementById('imageUpload');
imageUpload.disabled = true;

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri('https://raw.githubusercontent.com/daoshengmu/face-api-recognize/master/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('https://raw.githubusercontent.com/daoshengmu/face-api-recognize/master/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('https://raw.githubusercontent.com/daoshengmu/face-api-recognize/master/models')
]).then(init)

async function init() {
  const container = document.getElementById('container');
  const labeledFaceDescriptors = await loadLabeledImages();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
  status.innerHTML = "Loaded";
  imageUpload.disabled = false;

  let image;
  let canvas;
  imageUpload.addEventListener('change', async () => {
    if (canvas) image.remove();
    if (canvas) canvas.remove();

    status.innerHTML = "";  
    image = await faceapi.bufferToImage(imageUpload.files[0]);
    image.className = 'resize_fit_center';
    container.append(image);

    canvas = faceapi.createCanvasFromMedia(image);
    canvas.className = 'canvas_fit_center';
    container.append(canvas);

    const displaySize = {width: image.width, height: image.height};
    faceapi.matchDimensions(canvas, displaySize);
    const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors();
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor));

    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box
      const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
      drawBox.draw(canvas);
    })
  })
}

function loadLabeledImages() {
  const labels = ['curry', 'klay'];

  return Promise.all(
    labels.map(async label => {
      const descriptions = [];
      for (let i = 1; i <= 4; ++i) {
        const img = await faceapi.fetchImage(`https://raw.githubusercontent.com/daoshengmu/face-api-recognize/master/labeled_images/${label}/${i}.jpg`)
        const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
        descriptions.push(detections.descriptor);
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions)
    })
  )
}

window.onload = function() {
  document.getElementById('imageUpload').value = '';
}

