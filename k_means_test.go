package k_means

import (
	"testing"
	"math"
	"fmt"
	"io/ioutil"
	"strings"
	"strconv"
)

var randomOutput int = 0

type ComparableFloatSlice []float64

func (slice ComparableFloatSlice) Equals(other ComparableFloatSlice, threshold float64) bool {
	for i := 0; i < len(slice); i++ {
		if !Similar(slice[i], other[i], threshold) {
			return false
		}
	}
	return true
}

func Similar(a, b, threshold float64) bool {
	return math.Abs(a - b) < threshold
}

func TestNewKMeans(t *testing.T) {
	randomOutput = 0

	d := dataset()

	kmeans := NewKMeans(d, 2, random)

	firstCentroid := []float64 { 83, 81, 16, 98, 53 }
	secondCentroid := []float64 { 66, 49, 83, 15, 8 }

	if !ComparableFloatSlice(kmeans.centroids[0]).Equals(firstCentroid, 1e-16) {
		t.Errorf("Expected: %v, got: %v", firstCentroid, kmeans.centroids[0])
	}

	if !ComparableFloatSlice(kmeans.centroids[1]).Equals(secondCentroid, 1e-16) {
		t.Errorf("Expected: %v, got: %v", secondCentroid, kmeans.centroids[1])
	}
}

func TestKMeans_Fit(t *testing.T) {
	randomOutput = 0

	d := dataset()

	kmeans := NewKMeans(d, 2, random)
	kmeans.Fit(10)

	firstCentroid := []float64 { 72.33, 44.67, 28, 80.67, 70.33 }
	secondCentroid := []float64 { 45.67, 47.56, 72.11, 43.33, 40.44 }

	if !ComparableFloatSlice(kmeans.centroids[0]).Equals(firstCentroid, 1e-2) {
		t.Errorf("Expected: %v, got: %v", firstCentroid, kmeans.centroids[0])
	}

	if !ComparableFloatSlice(kmeans.centroids[1]).Equals(secondCentroid, 1e-2) {
		t.Errorf("Expected: %v, got: %v", secondCentroid, kmeans.centroids[1])
	}
}

/*

1 = 4,446,647,299
2 = 3,474,792,001
4 = 3,321,101,144
8 = 3,322,012,727

 */

func BenchmarkKMeans_Fit(b *testing.B) {
	d := loadIris()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		randomOutput = 0
		kmeans := NewKMeans(d, 2, random)
		kmeans.maxGoroutines = 8
		kmeans.Fit(10000)
	}

}

func random(a int) int {
	out := randomOutput
	randomOutput++

	return out
}

func dataset() [][]float64 {
	return [][]float64{
		{ 8, 78, 95, 91, 24, },
		{ 83, 81, 16, 98, 53, },
		{ 47, 23, 49, 98, 98, },
		{ 66, 49, 83, 15, 8, },
		{ 3, 32, 44, 29, 54, },
		{ 64, 20, 57, 49, 31, },
		{ 87, 30, 19, 46, 60, },
		{ 5, 23, 68, 42, 80, },
		{ 86, 96, 71, 20, 5, },
		{ 62, 7, 98, 74, 94, },
		{ 78, 92, 92, 62, 47, },
		{ 39, 31, 41, 8, 21, },
	}
}

func Test_setMapSafe(t *testing.T) {
	m := make(map[int] string)

	numberOfIterations := 32

	done := make(chan int)

	for i := 0; i < numberOfIterations; i++ {
		go func(p int) {
			setMapSafe(m, p, fmt.Sprint(p))
			done <- 1
		}(i)
	}

	for i := 0; i < numberOfIterations; i++ {
		<-done
	}

	for i := 0; i < numberOfIterations; i++ {
		actual := m[i]
		expected := fmt.Sprint(i)

		if actual != expected {
			t.Errorf("Actual: %v, expected: %v", actual, expected)
		}
	}
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func loadIris() Matrix {
	data, err := ioutil.ReadFile("data.csv")
	check(err)

	rows := strings.Split(string(data), "\n")[1:]

	var dataset Matrix

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

	return dataset
}

