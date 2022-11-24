package main

import (
	"log"
	"runtime"
	"time"
)

func PrintRam() {
	var meminfo runtime.MemStats
	runtime.ReadMemStats(&meminfo)
	log.Printf("RAM in MB: Heap=%v, Total=%v, Sys=%v", meminfo.Alloc/1024.0/1024.0,
		meminfo.TotalAlloc/1024.0/1024.0, meminfo.Sys/1024.0/1024.0)
}

func printRam(meminfo runtime.MemStats) {
	runtime.ReadMemStats(&meminfo)
	log.Printf("RAM in MB: Heap=%v, Total=%v, Sys=%v", meminfo.Alloc/1024.0/1024.0,
		meminfo.TotalAlloc/1024.0/1024.0, meminfo.Sys/1024.0/1024.0)
}

func PeriodicMemCheck(seconds time.Duration) {
	t := time.NewTicker(seconds * time.Second)
	var meminfo runtime.MemStats
	defer t.Stop()
	for {
		select {
		case <-t.C:
			printRam(meminfo)
		}
	}
}
