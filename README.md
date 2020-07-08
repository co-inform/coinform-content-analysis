# Co-Inform Content Analysis

This is one of module of Co-Inform platform developed by [WeST](https://west.uni-koblenz.de/). It performs textual analysis of posts. Currently it processes Twitter post.
This repository contains the declaration for the API of [coinform-content-analysis] (http://co-inform.informatik.uni-stuttgart.de/docs).

It has following features:
- Stance detection `app/routers/rumour_verif.py`
- Veracity estimation `app/routers/rumour_verif.py`

## Getting Started
- under `data` folder, you need to have `external` folder which contains neg/pos words, and `models` contains stance, veracity models
- run docker this [compose file](https://github.com/isspek/coinform-docker) to install required libraries
- set up your configuration in `config.ini`
- finally run docker compose file in this repository. 

## Developers
- to add new estimator, implement a function in `app/estimators/`
- make sure output fields for your estimator listed in `app/models/tweet.py`

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Ipek Baris** - *Initial work* - [isspek](https://github.com/isspek)

See also the list of [contributors](https://github.com/coinform-content-analysis/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments



