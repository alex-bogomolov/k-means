package main

import (
	"fmt"
	"github.com/alex-bogomolov/k-means"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"io/ioutil"
	"strconv"
	"strings"
	"time"
	"math/rand"
)

var source = rand.NewSource(time.Now().Unix())

func main() {
	run_kmeans()
}

func run_kmeans() {
	data, err := ioutil.ReadFile("data.csv")
	check(err)

	rows := strings.Split(string(data), "\n")[1:]

	var dataset [][]float64

	for _, row := range rows {
		if row == "" {
			continue
		}

		numbers := strings.Split(row, ",")

		x_1, err := strconv.ParseFloat(numbers[1], 64)
		x_2, err := strconv.ParseFloat(numbers[2], 64)
		check(err)

		dataset = append(dataset, []float64{x_1, x_2})
	}

	p, err := plot.New()
	check(err)

	p.Title.Text = "IRIS"
	p.X.Label.Text = "X1"
	p.Y.Label.Text = "X2"

	points := make(plotter.XYs, len(dataset))

	for index, point := range dataset {
		points[index].X = point[0]
		points[index].Y = point[1]
	}

	err = plotutil.AddScatters(p, "Points", points)
	check(err)

	err = p.Save(vg.Centimeter*15, vg.Centimeter*15, "dataset.png")
	check(err)


	kmeans := k_means.NewKMeans(dataset, 3, random)
	kmeans.SetMaxGoroutines(4)
	kmeans.Fit(10)

	results := kmeans.AssignPoints(dataset)

	resultPlot, err := plot.New()
	check(err)

	resultPlot.Title.Text = "IRIS"
	resultPlot.X.Label.Text = "X1"
	resultPlot.Y.Label.Text = "X2"

	var args []interface{}

	for cluster, points := range results {
		plotPoints := make(plotter.XYs, len(points))

		for index, point := range points {
			plotPoints[index].X = point[0]
			plotPoints[index].Y = point[1]
		}

		args = append(args, fmt.Sprintf("Cluster %d", cluster), plotPoints)

	}
	err = plotutil.AddScatters(resultPlot, args...)
	check(err)

	err = resultPlot.Save(vg.Centimeter*15, vg.Centimeter*15,
		"clusterized_dataset.png")
	check(err)
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func random (max int) int {
	r := rand.New(source)
	return r.Intn(max)
}

