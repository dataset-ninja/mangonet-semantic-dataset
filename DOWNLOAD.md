Dataset **MangoNet** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/supervisely-supervisely-assets-public/teams_storage/M/J/mJ/2oRdRXYBTUz1Kew63FW9SuHU4g1cyXdU8megr2FGggZyiM18XXu8WZn89YdpMJf8XMkn1AYbkEoKxb8XX7FIfN3mW1oBdQVIAO67V0hYUW9hf09kbcStms9LwF2t.tar)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='MangoNet', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

The data in original format can be [downloaded here](https://github.com/avadesh02/MangoNet-Semantic-Dataset/archive/refs/heads/master.zip).