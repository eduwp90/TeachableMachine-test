import logo from './logo.svg';
import './App.css';
import * as tf from '@tensorflow/tfjs';
import * as tmPose from '@teachablemachine/pose';
import Webcam from 'react-webcam';

import React, { useRef } from 'react';

const URL = 'https://teachablemachine.withgoogle.com/models/HQvC3rR8v/';
// const URL = 'https://teachablemachine.withgoogle.com/models/jwj-LGant/';
let model, webcam, ctx, labelContainer, maxPredictions;

function App() {
  const webcamRef = useRef(null);

  async function init() {
    const modelURL = URL + 'model.json';
    const metadataURL = URL + 'metadata.json';

    // load the model and metadata
    // Refer to tmImage.loadFromFiles() in the API to support files from a file picker
    // Note: the pose library adds a tmPose object to your window (window.tmPose)
    model = await tmPose.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();

    // Convenience function to setup a webcam
    // const size = 640;
    // const size1 = 480;
    // const flip = true; // whether to flip the webcam
    // webcam = new tmPose.Webcam(size, size1, flip); // width, height, flip
    // await webcam.setup({
    //   video: {
    //     aspectRatio: { exact: 1.2 },
    //     resizeMode: 'crop-and-scale',
    //   },
    //   image: {
    //     zoom: { exact: 0 },
    //   },
    // }); // request access to the webcam
    // await webcam.play();
    window.requestAnimationFrame(loop);

    // append/get elements to the DOM
    const canvas = document.getElementById('canvas');
    canvas.width = 640;
    canvas.height = 480;
    ctx = canvas.getContext('2d');
    labelContainer = document.getElementById('label-container');
    for (let i = 0; i < maxPredictions; i++) {
      // and class labels
      labelContainer.appendChild(document.createElement('div'));
    }
  }

  async function loop(timestamp) {
    console.log(webcamRef);
    // webcam.update(); // update the webcam frame
    await predict();
    window.requestAnimationFrame(loop);
  }

  async function predict() {
    console.log(webcamRef.current.getCanvas());
    // Prediction #1: run input through posenet
    // estimatePose can take in an image, video or canvas html element
    const { pose, posenetOutput } = await model.estimatePose(
      webcamRef.current.getCanvas()
    );
    // Prediction 2: run input through teachable machine classification model
    const prediction = await model.predict(posenetOutput);

    for (let i = 0; i < maxPredictions; i++) {
      const classPrediction =
        prediction[i].className + ': ' + prediction[i].probability.toFixed(2);
      labelContainer.childNodes[i].innerHTML = classPrediction;
    }

    // finally draw the poses
    drawPose(pose);
  }

  function drawPose(pose) {
    console.log(webcamRef);
    if (webcamRef.current.getCanvas()) {
      ctx.drawImage(webcamRef.current.getCanvas(), 0, 0);
      // draw the keypoints and skeleton
      if (pose) {
        const minPartConfidence = 0.5;
        tmPose.drawKeypoints(pose.keypoints, minPartConfidence, ctx);
        tmPose.drawSkeleton(pose.keypoints, minPartConfidence, ctx);
      }
    }
  }

  return (
    <>
      <div>Teachable Machine Pose Model</div>
      {/* <button type='button' onClick={() => init()}>
        Start
      </button> */}
      {webcamRef && (
        <Webcam
          ref={webcamRef}
          style={{
            position: 'absolute',
            marginLeft: 'auto',
            marginRight: 'auto',
            left: 0,
            right: 0,
            textAlign: 'center',
            zindex: -9,
            width: 640,
            height: 480,
          }}
          onUserMedia={init}
          mirrored={true}
        />
      )}
      <div>
        <canvas
          id='canvas'
          style={{
            position: 'absolute',
            marginLeft: 'auto',
            marginRight: 'auto',
            left: 0,
            right: 0,
            textAlign: 'center',
            zindex: 9,
            width: 640,
            height: 480,
          }}
        ></canvas>
      </div>
      <div id='label-container'></div>
    </>
  );
}

export default App;
