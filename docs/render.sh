# Note make sure this always uses the Linux line encoding :)
set -e
pandoc whitepaper.md -o whitepaper.pdf --bibliography whitepaper.bib --csl acm-sig-proceedings-long-author-list.csl

/mnt/c/Windows/System32/cmd.exe /C start whitepaper.pdf
#/mnt/c/Program\ Files\ \(x86\)/Google/Chrome/Application/chrome.exe whitepaper.pdf
#less whitepaper.pdf