PAPER := full-paper
LATEXMK := latexmk

.PHONY: all clean

all:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(PAPER).tex

clean:
	$(LATEXMK) -C $(PAPER).tex
	rm -f $(PAPER).bbl
