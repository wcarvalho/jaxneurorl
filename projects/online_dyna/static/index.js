document.addEventListener('DOMContentLoaded', function () {
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
      imgElement.src = data.image;
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