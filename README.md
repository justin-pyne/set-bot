# Set Card Game Bot
This is a project to create an automated bot that can identify all sets of cards from a visual input for the Set card game. The bot uses computer vision techniques like object detection and segmentation to isolate the cards, and applies image preprocessing to normalize the images. It then analyzes the cards to identify all possible sets, which can be displayed to the user or used to play the game automatically.

The bot can take input from a live video stream or from static images, and can be run on a local computer or on a server. It is implemented in Python using OpenCV and other popular computer vision libraries.

## Getting Started
To get started with the Set card game bot, you will need to have Python 3 installed on your computer. You will also need to install the required Python packages, which are listed in the requirements.txt file.

Once you have installed the required packages, you can run the bot by running the app.py script. This will open a window displaying the video stream or image, with bounding boxes drawn around the detected cards.
