package k_means

import (
	"math"
	"reflect"
	_ "runtime"
	"sync"
	//"runtime"
	"github.com/alex-bogomolov/functional_go"
	"runtime"
)

type RandomNumberGenerator func(int) int

type Matrix [][]float64

func (m Matrix) shape() (int, int) {
	return len(m), len(m[0])
}

type KMeans struct {
	points        Matrix
	clusters      int
	centroids     Matrix
	assignment    map[int]int
	random        RandomNumberGenerator
	maxGoroutines int
}

func NewKMeans(points Matrix, clusters int, generator RandomNumberGenerator) *KMeans {
	out := &KMeans{
		points:   points,
		clusters: clusters,
		random:   generator,
	}
	out.maxGoroutines = runtime.NumCPU()
	out.initCentroids()

	return out
}

func (kmeans *KMeans) Fit(maxIter int) {
	for i := 0; i < maxIter; i++ {
		kmeans.assignPoints()
		kmeans.moveCentroids()
	}
}

func (kmeans *KMeans) initCentroids() {
	var centroids []int
	kmeans.centroids = make([][]float64, kmeans.clusters, kmeans.clusters)

	for i := 0; i < kmeans.clusters; i++ {
		index := kmeans.random(len(kmeans.points))
		centroids = append(centroids, index)

		for contains(centroids, index) {
			index = kmeans.random(len(kmeans.points))
		}

		kmeans.centroids[i] = kmeans.points[index]
	}
}

func (kmeans KMeans) AssignPoints(points Matrix) map[int][][]float64 {
	assignment := make(map[int][][]float64)

	for _, point := range points {
		distances := make([]float64, kmeans.clusters, kmeans.clusters)

		for cIndex := 0; cIndex < kmeans.clusters; cIndex++ {
			centroid := kmeans.centroids[cIndex]

			distances[cIndex] = distance(point, centroid)
		}

		index := indexOfMin(distances)
		assignment[index] = append(assignment[index], point)
	}

	return assignment
}

func (kmeans *KMeans) assignPoints() {
	kmeans.assignment = make(map[int]int)
	semaphore := make(chan int, kmeans.maxGoroutines)
	done := make(chan int)

	for pointIndex := 0; pointIndex < len(kmeans.points); pointIndex++ {
		go func(pointIndex int) {
			semaphore <- 1
			point := kmeans.points[pointIndex]

			distances := make([]float64, kmeans.clusters, kmeans.clusters)

			for cIndex := 0; cIndex < kmeans.clusters; cIndex++ {
				centroid := kmeans.centroids[cIndex]

				distances[cIndex] = distance(point, centroid)
			}

			setMapSafe(kmeans.assignment, pointIndex, indexOfMin(distances))
			<-semaphore
			done <- 1
		}(pointIndex)
	}

	for pointIndex := 0; pointIndex < len(kmeans.points); pointIndex++ {
		<-done
	}
}

func (kmeans *KMeans) moveCentroids() {
	for centroidIndex := 0; centroidIndex < kmeans.clusters; centroidIndex++ {
		assignedPoints := functional_go.SelectWithIndex(kmeans.points, func(index int, _ []float64) bool {
			return kmeans.assignment[index] == centroidIndex
		}).(Matrix)

		newCentroid := average(assignedPoints)
		kmeans.centroids[centroidIndex] = newCentroid
	}
}

func average(points Matrix) []float64 {
	m, n := points.shape()

	out := make([]float64, n, n)

	for i := 0; i < n; i++ {
		col := 0.0

		for j := 0; j < m; j++ {
			col += points[j][i]
		}

		out[i] = col / float64(m)
	}

	return out
}

func contains(slice []int, number int) bool {
	for _, element := range slice {
		if element == number {
			return true
		}
	}
	return false
}

func (kmeans *KMeans) SetMaxGoroutines(max int) {
	kmeans.maxGoroutines = max
}

func distance(a, b []float64) float64 {
	sum := 0.0

	for i := 0; i < len(a); i++ {
		sum += math.Pow(a[i]-b[i], 2)
	}

	return math.Sqrt(sum)
}

func indexOfMin(slice []float64) int {
	min := slice[0]
	minIndex := 0

	for i := 1; i < len(slice); i++ {
		if slice[i] < min {
			min = slice[i]
			minIndex = i
		}
	}

	return minIndex
}

var mutex = sync.Mutex{}

func setMapSafe(m, key, value interface{}) {
	mutex.Lock()
	mValue := reflect.ValueOf(m)
	mValue.SetMapIndex(reflect.ValueOf(key), reflect.ValueOf(value))
	mutex.Unlock()
}
