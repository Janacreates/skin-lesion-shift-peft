PYTHON ?= python3

.PHONY: install download train-all report test shift attention export app

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

# v2: shift detectors across 5 seeds, then the deployable int8 model + detector, then the demo app
shift:
	$(PYTHON) -m skinshift.shift --all-seeds

# where the model looks: gradient-weighted attention maps, dermoscopy vs smartphone, 5 seeds
attention:
	$(PYTHON) -m skinshift.attention --all-seeds

export:
	$(PYTHON) -m skinshift.export

app:
	$(PYTHON) deploy/app.py
