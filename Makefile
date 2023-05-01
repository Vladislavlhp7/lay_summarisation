.PHONY: report

report:
	cd report
	pandoc -s -o report/report.pdf report/report.md