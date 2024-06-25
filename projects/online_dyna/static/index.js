async function loadTensorFlow() {
  const tf = await import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js');
  return tf;
}


function makeAgentTile(direction, tileSize) {
  const TRI_COORDS = [
    [0.12, 0.19],
    [0.87, 0.50],
    [0.12, 0.81]
  ];

  let agentTile = tf.zeros([tileSize, tileSize, 3]);

  agentTile = fillCoords(
    agentTile,
    pointInTriangle(...TRI_COORDS),
    [255, 0, 0]
  );


  if (direction === 0) {
    return agentTile;  // right
  } else if (direction === 1) {
    return rot90(agentTile, 3);  // down
  } else if (direction === 2) {
    return rot90(agentTile, 2);  // left
  } else if (direction === 3) {
    return rot90(agentTile, 1);  // up
  }
}

function fillCoords(img, fn, color) {
  return tf.tidy(() => {
    const shape = img.shape;
    const yCoords = tf.linspace(0.5 / shape[0], 1 - 0.5 / shape[0], shape[0]);
    const xCoords = tf.linspace(0.5 / shape[1], 1 - 0.5 / shape[1], shape[1]);
    const [xs, ys] = tf.meshgrid(xCoords, yCoords);

    const mask = tf.tidy(() => {
      const flatXs = xs.flatten();
      const flatYs = ys.flatten();
      const points = tf.stack([flatXs, flatYs], 1);
      const fnResults = points.arraySync().map(p => fn(p[0], p[1]));
      return tf.tensor(fnResults).reshape([shape[0], shape[1]]);
    });

    const colorTensor = tf.tensor(color);
    return tf.where(mask.expandDims(-1), colorTensor, img);
  });
}

function pointInTriangle(a, b, c) {
  return function (x, y) {
    const v0 = [c[0] - a[0], c[1] - a[1]];
    const v1 = [b[0] - a[0], b[1] - a[1]];
    const v2 = [x - a[0], y - a[1]];

    // Compute dot products
    const dot00 = v0[0] * v0[0] + v0[1] * v0[1];
    const dot01 = v0[0] * v1[0] + v0[1] * v1[1];
    const dot02 = v0[0] * v2[0] + v0[1] * v2[1];
    const dot11 = v1[0] * v1[0] + v1[1] * v1[1];
    const dot12 = v1[0] * v2[0] + v1[1] * v2[1];

    // Compute barycentric coordinates
    const invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
    const u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    const v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    // Check if point is in triangle
    return (u >= 0) && (v >= 0) && (u + v) < 1;
  };
}

function rot90(tensor, k = 1) {
  return tf.tidy(() => {
    k = ((k % 4) + 4) % 4; // Normalize k to be 0, 1, 2, or 3
    let result = tensor;
    for (let i = 0; i < k; i++) {
      result = tf.transpose(result.reverse(1), [1, 0, 2]);
    }
    return result;
  });
}

//function createImageFromGrid(grid, agentPos, agentDir, imageDict, includeObjects = true) {
//  // Assumes wall_index is the index for the wall image in imageDict['images']

//  let wallIndex = imageDict.keys.indexOf('wall');

//  // Expand grid size by 2 in each direction (top, bottom, left, right)

//  let H = grid.shape[0];
//  let W = grid.shape[1];
//  let newH = H + 2;
//  let newW = W + 2;

//  // Create a new grid with wall index
//  let newGrid = tf.fill([newH, newW], wallIndex);

//  // Place the original grid in the center of the new grid
//  newGrid = newGrid.arraySync(); // Convert tensor to array for easier manipulation
//  let gridArray = grid.arraySync();
//  for (let i = 0; i < H; i++) {
//    for (let j = 0; j < W; j++) {
//      newGrid[i + 1][j + 1] = gridArray[i][j];
//    }
//  }

//  // Flatten the grid for easier indexing
//  newGrid = tf.tensor(newGrid);
//  let newGridFlat = newGrid.flatten();

//  // Retrieve the images tensor
//  let images = tf.tensor(imageDict.images);

//  // Use advanced indexing to map the grid indices to actual images
//  if (!includeObjects) {
//    newGridFlat = newGridFlat.where(newGridFlat.greater(1), tf.zerosLike(newGridFlat));
//  }

//  newGridFlat = newGridFlat.arraySync();
//  let flatImages = newGridFlat.map(index => images.slice([index, 0, 0], [1, -1, -1]).arraySync()[0]);

//  // Create the agent tile with the specified direction
//  let tileSize = images.shape[1];
//  let agentTile = makeAgentTile(agentDir, tileSize).arraySync();

//  // Adjust agent position to account for the expanded grid
//  let agentY = agentPos[0] + 1;
//  let agentX = agentPos[1] + 1;

//  // Dimensions of the new grid
//  let imgH = images.shape[1];
//  let imgW = images.shape[2];
//  let C = images.shape[3];

//  // Reshape and transpose to form the single image
//  let reshapedImages = ndarray(new Float32Array(flatImages.flat()), [newH, newW, imgH, imgW, C]);

//  // Set agent tile at the agent's position
//  for (let i = 0; i < tileSize; i++) {
//    for (let j = 0; j < tileSize; j++) {
//      for (let c = 0; c < C; c++) {
//        reshapedImages.set(agentY, agentX, i, j, c, agentTile[i][j][c]);
//      }
//    }
//  }

//  // Transpose to (new_H, img_H, new_W, img_W, C) and reshape to (new_H * img_H, new_W * img_W, C)
//  let transposedImages = ndarray(new Float32Array(newH * newW * imgH * imgW * C), [newH, imgH, newW, imgW, C]);

//  for (let nH = 0; nH < newH; nH++) {
//    for (let iH = 0; iH < imgH; iH++) {
//      for (let nW = 0; nW < newW; nW++) {
//        for (let iW = 0; iW < imgW; iW++) {
//          for (let c = 0; c < C; c++) {
//            transposedImages.set(nH, iH, nW, iW, c, reshapedImages.get(nH, nW, iH, iW, c));
//          }
//        }
//      }
//    }
//  }

//  let finalImage = tf.tensor(transposedImages.transpose(0, 2, 1, 3, 4).reshape([newH * imgH, newW * imgW, C]));

//  return finalImage;
//}

function createImageFromGrid(grid, agentPos, agentDir, imageDict, includeObjects = true) {
  const wallIndex = imageDict.keys.indexOf('wall');
  const [H, W, D] = grid.shape;
  const newH = H + 2, newW = W + 2;

  // Place the original grid in the center of the new grid
  const newGrid = tf.pad(grid, [[1, 1], [1, 1], [0, 0]], wallIndex);  // Pad with -1

  // Retrieve the images tensor
  const images = tf.tensor(imageDict.images);
  const [numImages, imgH, imgW, C] = images.shape;

  // If not including objects, set all object indices to 0
  if (!includeObjects) {
    newGrid = newGrid.where(newGrid.greater(1), tf.scalar(0));
  }

  // Convert the grid indices to actual images
  let finalImage = tf.tidy(() => {
    const flatGrid = newGrid.reshape([-1, D]);
    const mappedImages = tf.gather(images, flatGrid.cast('int32'));
    return mappedImages.reshape([newH, newW, imgH, imgW, C]);
  });

  // Create the agent tile with the specified direction
  const agentTile = makeAgentTile(agentDir, imgH, imgW, C);

  // Adjust agent position to account for the expanded grid
  let [agentY, agentX] = agentPos;
  agentX += 1;
  agentY += 1;

  finalImage = tf.tidy(() => {
    const finalImageArray = finalImage.arraySync();
    const agentTileArray = agentTile.arraySync();

    // Set the agent tile at the specified position
    finalImageArray[agentY][agentX] = agentTileArray;

    return tf.tensor(finalImageArray);
  });

  // Transpose the image
  finalImage = finalImage.transpose([0, 2, 1, 3, 4]);

  // Reshape
  finalImage = finalImage.reshape([newH * imgH, newW * imgW, C]);

  return finalImage;
}

function tensorToBase64Image(tensor) {
  return new Promise((resolve, reject) => {
    tf.tidy(() => {
      try {
        const [height, width, channels] = tensor.shape;

        if (channels !== 3) {
          throw new Error('Input tensor must have 3 channels (RGB)');
        }

        // Normalize the tensor values to 0-255 and convert to Uint8Array
        const normalizedData = tf.clipByValue(tensor, 0, 255).toInt().dataSync();

        // Create canvas
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');

        // Create ImageData
        const imageData = ctx.createImageData(width, height);

        // Set RGB data and add alpha channel
        for (let i = 0, j = 0; i < normalizedData.length; i += 3, j += 4) {
          imageData.data[j] = normalizedData[i];     // R
          imageData.data[j + 1] = normalizedData[i + 1]; // G
          imageData.data[j + 2] = normalizedData[i + 2]; // B
          imageData.data[j + 3] = 255;                   // A (fully opaque)
        }

        // Put the image data on the canvas
        ctx.putImageData(imageData, 0, 0);

        // Convert to base64
        const base64Image = canvas.toDataURL('image/png');
        resolve(base64Image);
      } catch (error) {
        reject(error);
      }
    });
  });
}

// Function to convert tensor to base64 image
async function tensorToBase64(tensor) {
  // Convert tensor to Uint8Array
  const [height, width, channels] = tensor.shape;
  const tensorData = await tensor.data();
  const uint8Array = new Uint8Array(tensorData);

  // Create an offscreen canvas
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  // Create ImageData and put it in the canvas context
  const imageData = new ImageData(uint8Array, width, height);
  ctx.putImageData(imageData, 0, 0);

  // Convert canvas to base64
  return canvas.toDataURL();
}


document.addEventListener('DOMContentLoaded', async function () {
  //const tf = await loadTensorFlow();

  document.addEventListener('keydown', function (event) {
    switch (event.key) {
      case "ArrowUp":
      case "ArrowDown":
      case "ArrowLeft":
      case "ArrowRight":
        event.preventDefault();
        break;
      default:
        break;
    }
  });
  // Create a global variable to store the image data
  var imageData = null;

  var socket = io();

  // Object to store event times
  var eventTimes = {
    imageSeenTimes: []
  };

  function getLatestTime(timesArray) {
    // Returns the latest time from the times array or 'undefined' if empty
    return timesArray.length > 0 ? timesArray[timesArray.length - 1] : undefined;
  }

  /////////////////////////
  // Handling left and right clicks
  /////////////////////////
  // Function to emit an event to the server when the arrows are clicked
  function recordClick(direction) {
    socket.emit('record_click', { direction: direction });
    console.log(direction)
  }
  // Store the event handlers in a global variable for reuse
  var arrowListeners = {
    left: function () {
      recordClick('left');
      console.log('left arrow');
    },
    right: function () {
      recordClick('right');
      console.log('right arrow');
    }
  };

  // Function to remove previous listeners and add new ones
  function reinitializeArrowListeners() {
    var leftArrow = document.getElementById('left-arrow');
    if (leftArrow) {
      // Remove old listener
      leftArrow.removeEventListener('click', arrowListeners.left);
      // Add new listener
      leftArrow.addEventListener('click', arrowListeners.left);
    }

    var rightArrow = document.getElementById('right-arrow');
    if (rightArrow) {
      // Remove old listener
      rightArrow.removeEventListener('click', arrowListeners.right);
      // Add new listener
      rightArrow.addEventListener('click', arrowListeners.right);
    }
  }

  //// Left and right arrow click listeners
  //var leftArrow = document.getElementById('left-arrow');
  //if (leftArrow) {
  //  leftArrow.addEventListener('click', function () {
  //    recordClick('left');
  //    console.log('left arrow')
  //  });
  //}

  //var rightArrow = document.getElementById('right-arrow');
  //if (rightArrow) {
  //  rightArrow.addEventListener('click', function () {
  //    recordClick('right');
  //    console.log('right arrow')
  //  });
  //}

  /////////////////////////
  // Loading content as soon as sock connects (used in initial connect)
  /////////////////////////
  // Request dynamic content updates once the WebSocket connection is established
  socket.on('connect', function () {
    socket.emit('request_update');  // Request the dynamic content updates as soon as the connection is established
  });

  /////////////////////////
  // Loading image data
  /////////////////////////
  // Request dynamic content updates once the WebSocket connection is established
  socket.on('load_data', function (data) {
    // Save the image data to the global variable
    imageData = data.image_data;
    console.log('Image data received:', imageData);
  });

  // Function to access the image data
  function getImageData() {
    return imageData;
  }

  /////////////////////////
  // Change content in main html
  /////////////////////////
  // Listen for the 'update_content' event from the server to update the page content
  socket.on('update_content', function (data) {
    document.getElementById('content').innerHTML = data.content;
  });

  /////////////////////////
  // Change aspects of stage's html
  /////////////////////////
  // Listen for the 'update_html_fields' event from the server to update the page content
  socket.on('update_html_fields', function (data) {
    var title = document.getElementById('title')
    console.log("========================")
    console.log('STAGE: ' + data.title)
    if (title){
      document.getElementById('title').innerHTML = data.title
      document.getElementById('subtitle').innerHTML = data.subtitle
      document.getElementById('body').innerHTML = data.body
    }
    var taskDesc = document.getElementById('taskDesc');
    if (taskDesc) {
      taskDesc.innerHTML = data.taskDesc
    }
    var envcaption = document.getElementById('envcaption');
    if (envcaption) {
      envcaption.innerHTML = data.envcaption
    }
    reinitializeArrowListeners();
  });

  /////////////////////////
  // Key press event
  /////////////////////////
  document.addEventListener('keydown', function (event) {
    // Record the current time when the keydown event occurs
    var keydownTime = new Date(); // Get the current time
    var imageSeenTime = getLatestTime(eventTimes.imageSeenTimes);

    // Emit the keydown event with the latest times
    socket.emit('key_pressed', {
      key: event.key,
      keydownTime: keydownTime,
      imageSeenTime: imageSeenTime
    });
  });

  /////////////////////////
  // Environment action taken event
  /////////////////////////
  socket.on('action_taken', function (data) {

    // Record the current time when the action_taken event occurs
    var currentTime = new Date(); // Get the current time
    eventTimes.imageSeenTimes.push(currentTime); // Store it in the array

    var imgElement = document.getElementById('stateImage');
    if (imgElement) {
      if (data.image) {
        imgElement.src = data.image;
      } else {
        // Create the final image tensor
        //imageData = getImageData()
        var grid = tf.tensor(data.state.grid)
        const finalImage = createImageFromGrid(
          grid,
          data.state.agent_pos,
          data.state.agent_dir,
          imageData);

        // Convert the tensor to a base64 image and set it to the img element
        tensorToBase64Image(finalImage)
          .then(base64Image => {
            imgElement.src = base64Image;
          })
          .catch(error => {
            console.error('Error converting tensor to base64 image:', error);
          })
          .finally(() => {
            // Make sure to dispose of the tensors when you're done
            finalImage.dispose();
            grid.dispose();
          });
      }
    }

    /////////////////////////
    // Stage of environment changed
    /////////////////////////
    socket.on('stage_changed', function (data) {
      var contentElement = document.getElementById('content');
      if (contentElement) {
        contentElement.innerHTML = '<p>' + data.stage_text + '</p>';
      }
      //var imgElement = document.getElementById('stateImage');
      //if (imgElement) {
      //  imgElement.src = data.image;
      //}
    });
  });

  /////////////////////////
  // Add timer?
  /////////////////////////
  socket.on('start_timer', function (data) {
    var timerDuration = data.seconds; // Set the timer duration in seconds
    var timerElement = document.getElementById('timer');
    console.log('timer started')

    // Display the initial timer value
    timerElement.textContent = 'Time remaining: ' + timerDuration + ' seconds';

    // Update the timer every second
    var timerInterval = setInterval(function () {
      timerDuration--;
      timerElement.textContent = 'Time remaining: ' + timerDuration + ' seconds';
      console.log('time remaining: ' + timerDuration + ' seconds')

      if (timerDuration <= 0) {
        clearInterval(timerInterval);
        socket.emit('timer_finished'); // Emit the 'timer_finished' event to the server
        console.log('timer finished')
      }
    }, 1000);
  });

  // Listen for the 'stage_advanced' event
  socket.on('stop_timer', function () {
    if (typeof timerInterval !== 'undefined') { // Check if timerInterval is defined
      clearInterval(timerInterval); // Clear the timer interval
      console.log('stop timer')
    }
    var timer = document.getElementById('timer')
    if (timer) {
      timer.textContent = ''; // Clear the timer display
      console.log('remove timer content')
    }
  });
});