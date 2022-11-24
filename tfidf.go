package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/go-nlp/tfidf"
	"github.com/xtgo/set"
	"gorgonia.org/tensor"
)

func ReadCsv(path string, skipRows int) ([][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Opening '%s' file as CSV", path)
	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		message := fmt.Sprintf("Unable to parse CSV from '%s': %v", path, err)
		log.Printf(message)
		return nil, errors.New(message)
	}
	if skipRows > 0 {
		log.Printf("Skipping %d first rows from '%s' %d rows lenght", skipRows, path, len(records))
		records = records[skipRows:]
	} else {
		log.Printf("Read %d rows from '%s'", len(records), path)
	}
	return records, nil
}

func Tmp(records [][]string, query string) {
	corpus, invCorpus := makeCorpus(mobydick)
	docs := makeDocuments(mobydick, corpus)
	tf := tfidf.New()
	for _, doc := range docs {
		tf.Add(doc)
	}
	tf.CalculateIDF()
	fmt.Println("IDF:")
	for i, w := range invCorpus {
		fmt.Printf("\t%q: %1.1f\n", w, tf.IDF[i])
		if i >= 10 {
			break
		}
	}
}

var mobydick = []string{
	"some",
}

func Example() {
	corpus, invCorpus := makeCorpus(mobydick)
	docs := makeDocuments(mobydick, corpus)
	tf := tfidf.New()

	for _, doc := range docs {
		tf.Add(doc)
	}
	tf.CalculateIDF()

	fmt.Println("IDF:")
	for i, w := range invCorpus {
		fmt.Printf("\t%q: %1.1f\n", w, tf.IDF[i])
		if i >= 10 {
			break
		}
	}

	// now we search

	// "ishmael" is a query
	ishmael := doc{corpus["ishmael"]}

	// "whenever i find" is another query
	whenever := doc{corpus["whenever"], corpus["i"], corpus["find"]}

	// step1: score the queries
	ishmaelScore := tf.Score(ishmael)
	wheneverScore := tf.Score(whenever)

	// step2: find the docs that contains the queries.
	// if there are no docs, oops.
	ishmaelDocs, ishmaelRelVec := contains(ishmael, docs, tf)
	wheneverDocs, wheneverRelVec := contains(whenever, docs, tf)

	// step3: calculate the cosine similarity
	ishmaelRes := cosineSimilarity(ishmaelScore, ishmaelDocs, ishmaelRelVec)
	wheneverRes := cosineSimilarity(wheneverScore, wheneverDocs, wheneverRelVec)

	// step4: sort the results
	sort.Sort(sort.Reverse(ishmaelRes))
	sort.Sort(sort.Reverse(wheneverRes))

	fmt.Printf("Relevant Docs to \"Ishmael\":\n")
	for _, d := range ishmaelRes {
		fmt.Printf("\tID   : %d\n\tScore: %1.3f\n\tDoc  : %q\n", d.id, d.score, mobydick[d.id])
	}
	fmt.Println("")
	fmt.Printf("Relevant Docs to \"whenever i find\":\n")
	for _, d := range wheneverRes {
		fmt.Printf("\tID   : %d\n\tScore: %1.3f\n\tDoc  : %q\n", d.id, d.score, mobydick[d.id])
	}
}

type doc []int

func (d doc) IDs() []int { return []int(d) }

func makeDocuments(a []string, c map[string]int) []tfidf.Document {
	retVal := make([]tfidf.Document, 0, len(a))
	for _, s := range a {
		var ts []int
		for _, f := range strings.Fields(s) {
			f = strings.ToLower(f)
			id := c[f]
			ts = append(ts, id)
		}
		retVal = append(retVal, doc(ts))
	}
	return retVal
}

func makeCorpus(a []string) (map[string]int, []string) {
	retVal := make(map[string]int)
	invRetVal := make([]string, 0)
	var id int
	for _, s := range a {
		for _, f := range strings.Fields(s) {
			f = strings.ToLower(f)
			if _, ok := retVal[f]; !ok {
				retVal[f] = id
				invRetVal = append(invRetVal, f)
				id++
			}
		}
	}
	return retVal, invRetVal
}

type docScore struct {
	id    int
	score float64
}

type docScores []docScore

func (ds docScores) Len() int           { return len(ds) }
func (ds docScores) Less(i, j int) bool { return ds[i].score < ds[j].score }
func (ds docScores) Swap(i, j int) {
	ds[i].score, ds[j].score = ds[j].score, ds[i].score
	ds[i].id, ds[j].id = ds[j].id, ds[i].id
}

func cosineSimilarity(queryScore []float64, docIDs []int, relVec []float64) docScores {
	// special case
	if len(docIDs) == 1 {
		// even more special case!
		if len(queryScore) == 1 {
			return docScores{
				{docIDs[0], queryScore[0] * relVec[0]},
			}
		}

		q := tensor.New(tensor.WithBacking(queryScore))
		m := tensor.New(tensor.WithBacking(relVec))
		score, err := q.Inner(m)
		if err != nil {
			panic(err)
		}
		return docScores{
			{docIDs[0], score.(float64)},
		}
	}

	m := tensor.New(tensor.WithShape(len(docIDs), len(queryScore)), tensor.WithBacking(relVec))
	q := tensor.New(tensor.WithShape(len(queryScore)), tensor.WithBacking(queryScore))
	dp, err := m.MatVecMul(q)
	if err != nil {
		panic(err)
	}

	m2, err := tensor.Square(m)
	if err != nil {
		panic(err)
	}

	normDocs, err := tensor.Sum(m2, 1)
	if err != nil {
		panic(err)
	}

	normDocs, err = tensor.Sqrt(normDocs)
	if err != nil {
		panic(err)
	}

	q2, err := tensor.Square(q)
	if err != nil {
		panic(err)
	}
	normQt, err := tensor.Sum(q2)
	if err != nil {
		panic(err)
	}
	normQ := normQt.Data().(float64)
	normQ = math.Sqrt(normQ)

	norms, err := tensor.Mul(normDocs, normQ)
	if err != nil {
		panic(err)
	}

	cosineSim, err := tensor.Div(dp, norms)
	if err != nil {
		panic(err)
	}
	csData := cosineSim.Data().([]float64)

	var ds docScores
	for i, id := range docIDs {
		score := csData[i]
		ds = append(ds, docScore{id: id, score: score})
	}
	return ds
}
func contains(query tfidf.Document, in []tfidf.Document, tf *tfidf.TFIDF) (docIDs []int, relVec []float64) {
	q := query.IDs()
	q = set.Ints(q) // unique words only
	for i := range in {
		doc := in[i].IDs()

		var count int
		var relevant []float64
		for _, wq := range q {
		inner:
			for _, wd := range doc {
				if wq == wd {
					count++
					break inner
				}
			}
		}
		if count == len(q) {
			// calculate the score of the doc
			score := tf.Score(in[i])
			// get the  scores of the relevant words
			for _, wq := range q {
			inner2:
				for j, wd := range doc {
					if wd == wq {
						relevant = append(relevant, score[j])
						break inner2
					}
				}
			}
			docIDs = append(docIDs, i)
			relVec = append(relVec, relevant...)
		}
	}
	return
}
