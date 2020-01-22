#
# Makefile
# Shlomi Fish, 2020-01-22 21:06
#

all:
	@echo "Makefile needs your attention"

PYPROG = 685-v1.py

update: /home/shlomif/progs/riddles/project-euler/git/project-euler/685/685-v1.py
	cp -f $< $(PYPROG)

# vim:ft=make
#
