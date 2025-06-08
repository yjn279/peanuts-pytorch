# PEANUTS
"peanuts" is for building variations of PhaseNet.


## Installation
### Miniconda
```
conda create -n peanuts python=3.9
conda activate peanuts
conda install poetry
poetry install <commands>  # 下記参照
poetry run task setup
```

### venv（非推奨）
```
python3 -m venv venv
. venv/bin/activate
pip install poetry
poetry install <commands>  # 下記参照
poetry run task setup
```

`poetry install`では、使用する環境に合わせて以下のようにコマンドを付け、依存関係をインストールする。  
```
poetry install --extras macos --no-dev
```

- OS
    - Windows・Intel Mac・Linux：`--extras none`
    - Apple silicon：`--extras macos`

- Usage
    - User（書いたコードをpushしない人）：`--no-dev`
    - Developer（書いたコードをpushする人）：コマンドなし
    
    
## Getting Started
### Train
```
poetry run train
```

### Predict
```
poetry run predict
```

### Lint & Format (for developers)
```
poetry run task check
```


## .env
下記を参考に、`.env.sample`ファイルにならって`.env`ファイルを作成

### SLACK_WEBHOOK_URL (Optional)
1. [こちら](https://my.slack.com/services/new/incoming-webhook/)のページに移動
2. チャンネル（Slackbotがおすすめ）を選択してインテグレーションを追加
3. Webhook URLをコピーして設定を保存
4. 取得したURLを`.env`ファイルにペースト


## Learn More
### TensorFlow
- [TensorFlow](https://www.tensorflow.org/?hl=ja)
- [Tensorflow Plugin - Metal - Apple Developer](https://developer.apple.com/metal/tensorflow-plugin/)
### Poetry (Package manager)
- [Poetry: Python packaging and dependency management made easy](https://github.com/python-poetry/poetry)
- [Poetryをサクッと使い始めてみる](https://qiita.com/ksato9700/items/b893cf1db83605898d8a)


### Flake8 (Linter)
- [Flake8: Your Tool For Style Guide Enforcement](https://flake8.pycqa.org/en/latest/)
- [Pythonのコードフォーマッターについての個人的ベストプラクティス](https://qiita.com/sin9270/items/85e2dab4c0144c79987d)
- [flake8の設定をpyproject.tomlに統合する](https://qiita.com/que9/items/7cf9f992b8decb4265c1)

### Black (Formatter)
- [Black 22.10.0 documentation](https://black.readthedocs.io/en/stable/)
- [Pythonのコードフォーマッターについての個人的ベストプラクティス](https://qiita.com/sin9270/items/85e2dab4c0144c79987d)

### GitHub Actions (CI/CD)
- [GitHub Actionsのドキュメント](https://github.co.jp/features/actions)
- [Python Tips: Python のプロジェクトで GitHub Actions を使いたい](https://www.lifewithpython.com/2020/04/python-github-actions.html)
- [Pythonでの開発・CI/CDの私的ベストプラクティス2022](https://zenn.dev/dajiaji/articles/e29005502a9e78)

### Others
- [ObsPy](https://docs.obspy.org/)
- [Isort](https://pycqa.github.io/isort/)
