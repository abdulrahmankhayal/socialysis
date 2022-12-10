# socialysis
[![](https://img.shields.io/pypi/v/socialysis)](https://pypi.org/project/socialysis/)
[![PyPI - License](https://img.shields.io/pypi/l/socialysis)](LICENSE)

socialysis is a Python package for analyzing and visualizing Facebook Messenger data. It provides tools for creating and customizing various types of charts and graphs from your Messenger data, such as bar charts, line charts, pie charts, sunburst charts, dot charts, and gantt charts. With socialysis, you can easily gain insights from your Messenger data, including messages, media, links, reacts, photos, audio records, videos, emoji, words and more.

## Installation

    `pip install socialysis`
    
## Getting Started

To use `socialysis`, you first need to download your Facebook Messenger data from Facebook.
To download your Facebook Messenger data, follow these steps:

1. Go to your Facebook settings by clicking the dropdown arrow in the top right corner of the page, and selecting "Settings" from the dropdown menu.
2. In the left-hand menu, click on the "Your Facebook Information" option.
3. In the "Your Facebook Information" section, click on the "Download Your Information" option.
4. In the "Download Your Information" window, select the "Deselect All" button to deselect all data categories.
5. Scroll down to the "Messages" category and click on the "Messages" checkbox to select it.
6. In the "Format" dropdown menu, select the "JSON" option.
7. In the "Media Quality" dropdown menu, select the "Low" option.
8. Click on the "Create File" button to start the download process.
9. Once the download is complete, you will receive a notification and an email with a link to download the file. Click on the download link to download the file to your computer.
10. Extract the downloaded file to obtain the JSON files containing your Messenger data.

### Sample Data

Socialysis includes a sample data set that you can use to get started without needing to download your own Facebook data. To access the sample data, you can use the `restore` parameter when initializing the `Stats` class and set it to `'sample'`. For example:
  ```python
  from socialysis import Stats
  stats = Stats(restore='sample')
  ```
You can then use the `Stats` class and its attributes and modules to analyze the sample data.

## Usage
  See [Demos](./Demos) for detailed examples with explanations.
  
## Gallery

![Collection](/Gallery/Collections_3_2.png)

