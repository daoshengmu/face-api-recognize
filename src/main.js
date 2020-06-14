
const imageUpload = document.getElementById('imageUpload');

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(init)

async function init() {
  const container = document.createElement('div');
  container.style.position = 'relative';
  document.body.append(container);
  document.body.append('Loaded');
  const labeledFaceDescriptors = await loadLabeledImages();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

  let image;
  let canvas;
  imageUpload.addEventListener('change', async () => {
    if (canvas) image.remove();
    if (canvas) image.remove();

    image = await faceapi.bufferToImage(imageUpload.files[0]);
    container.append(image);
    canvas = faceapi.createCanvasFromMedia(image);
    container.append(canvas);
    const displaySize = {width: image.width, height: image.height};
    faceapi.matchDimensions(canvas, displaySize);
    const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors();
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    const result = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor));

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
        descriptions.push(detections.descriptor)
      }
    })
  )
}

window.onload = function() {
  document.getElementById('imageUpload').value = '';
}

