// main.go
package main

import (
	"log"

	"github.com/pocketbase/pocketbase"
)

func main() {
	records, err := ReadCsv("am_banks_conditions.csv", 2)
	if err != nil {
		log.Fatal(err)
	}
	Tmp(records, "")

	PrintRam()
	go PeriodicMemCheck(5)

	app := pocketbase.New()
	if err := app.Start(); err != nil {
		log.Fatal(err)
	}
}
