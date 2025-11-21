// SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
//
// SPDX-License-Identifier: MPL-2.0

let socket;
let imageElement;
let statusElement;
let frameCount = 0;
let lastFrameTime = Date.now();

/**
 * Initialize the Socket.IO connection and DOM elements
 */
document.addEventListener('DOMContentLoaded', () => {
    // Get references to DOM elements
    imageElement = document.getElementById('webcamImage');
    statusElement = document.getElementById('status');
    
    // Initialize Socket.IO connection
    initSocketIO();
});

/**
 * Set up Socket.IO connection and event handlers
 */
function initSocketIO() {
    // Connect to the server
    socket = io(`http://${window.location.host}`);
    
    // Connection successful
    socket.on('connect', () => {
        console.log('Connected to server');
        statusElement.textContent = 'Connected';
        statusElement.className = 'status connected';
    });
    
    // Receive webcam frames
    socket.on('webcam_frame', (data) => {
        // Update the image with the new frame
        imageElement.src = `data:image/jpeg;base64,${data.image}`;
        
        // Update frame rate display
        updateFrameRate();
        
        // Hide status once we start receiving frames
        if (frameCount === 1) {
            statusElement.style.display = 'none';
        }
    });
    
    // Connection lost
    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        statusElement.textContent = 'Connection lost';
        statusElement.className = 'status disconnected';
        statusElement.style.display = 'block';
    });
    
    // Connection error
    socket.on('connect_error', (error) => {
        console.error('Connection error:', error);
        statusElement.textContent = 'Connection error';
        statusElement.className = 'status error';
    });

    // Webcam error
    socket.on('webcam_error', (data) => {
        console.error('Webcam error:', data.error);
        statusElement.textContent = data.error;
        statusElement.className = 'status error';
        statusElement.style.display = 'block';
    });
}

/**
 * Calculate and log the actual frame rate
 */
function updateFrameRate() {
    frameCount++;
    
    // Calculate FPS every 30 frames
    if (frameCount % 30 === 0) {
        const now = Date.now();
        const elapsed = (now - lastFrameTime) / 1000; // Convert to seconds
        const fps = 30 / elapsed;
        console.log(`Receiving ~${fps.toFixed(1)} fps`);
        lastFrameTime = now;
    }
}