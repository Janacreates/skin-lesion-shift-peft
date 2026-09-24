PYTHON ?= python3

.PHONY: install download train-all report test

install:
	$(PYTHON) -m pip install -r requirements.txt

# pulls the class-balanced image subset from the ISIC Archive API (small: tens of MB)
download:
	$(PYTHON) -m skinshift.download

train-all:
	$(PYTHON) -m skinshift.train --method linear_probe
	$(PYTHON) -m skinshift.train --method full
	$(PYTHON) -m skinshift.train --method lora
	$(PYTHON) -m skinshift.report

test:
	$(PYTHON) -m pytest -q
