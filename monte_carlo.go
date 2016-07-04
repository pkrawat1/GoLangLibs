// Note the Library is a Portfolio Specific Library, Should Not be used on Schemes
// Monte Carlo Simulation Algorithm :
// The inputs to the process are:
// 1) Start capital
// 2) Annual mean return taken from the (10 portfolios)
// 3) Annual mean standard deviation
// 4) Number of years (observations)
// 5) Number of simulations = 500 => This is what is configurable 

//
// Logic:
//
// 1) Generate monthly mean return = annual return / 12
// 2) Generate monthly standard deviation = annual standard deviation / (square root 12)
// 3) Generate N random numbers where N = (number of years * number of simulations) with mean = mean return, std = standard deviation in a matrix
// Which has number of years as rows and number of simulations as columns
//
// Investment.matrix[number of years, number of simulation]
//
// 4) Define NAV as another matrix with number of years as rows and number of simulations as columns
//
// Loop:
// For i = 1: number of years
// NAV (i+1) = NAV (i) *  Investment.matrix[i,]

package main

import (
    "fmt"
    "math"
    "math/rand"
    "time"
    "sync"
    "runtime"
    "sort"
)

type MonteCarlo struct {
    GraphType string
    StartCapital float64
    AnnualMeanReturn float64
    AnnualReturnStandardDeviation float64
    Years int
    Time time.Time

    // Config Parameters
    NumberOfSimulations int
    AnnualInflation float64
    AnnualInflationStandardDeviation float64
    MonthlyWithdrawals float64

    MonthlyMeanReturn float64
    MonthlyRetStdDev float64
    MonthlyInflation float64
    MonthlyInfStdDev float64

    MonthlyInvestmentReturns [][]float64
    MonthlyInflationReturns [][]float64
    CombinedMatrix [][]float64
    StartCapitalMatrix [][]float64
    NAVMatrix [][]float64

    // Scenario Ranges
    GoodRange []float64
    AverageRange []float64
    BadRange []float64

    NumberOfObservations int
}

var MoCa MonteCarlo
var Result = make(map[string][][]interface{})
var wg sync.WaitGroup

func init() {
    runtime.GOMAXPROCS(runtime.NumCPU())

    // Dummy initialization
    MoCa = MonteCarlo{
        Time:                               time.Now(),
        GraphType:                          "yearly",
        StartCapital:                       10000,
        AnnualMeanReturn:                   0.1,
        AnnualReturnStandardDeviation:      0.1,
        Years:                              5,
        NumberOfSimulations:                500,
        AnnualInflation:                    0.025,
        AnnualInflationStandardDeviation:   0.015,
        MonthlyWithdrawals:                 0.0,
        GoodRange:                          []float64{0.8, 1.0},
        AverageRange:                       []float64{0.4, 0.7},
        BadRange:                           []float64{0.0, 0.3},
    }
}

func (MoCa *MonteCarlo) prediction() {
    MoCa.convert_annual_values_to_monthly_values()
    MoCa.calculate_number_of_observations()

    wg.Add(2)
    go MoCa.simulate_monthly_invest_return()
    go MoCa.simulate_monthly_inflation_return()
    wg.Wait()

    wg.Add(2)
    go MoCa.generate_combined_random_matrix()
    go MoCa.generate_matrix_for_start_capital()
    wg.Wait()
    
    MoCa.simulate_nav_values_for_portfolio()
    MoCa.replace_all_values_less_than_zero_to_mean_of_row_in_nav_matrix()

    MoCa.generate_scenario_distribution()
}

func main() {
    fmt.Println("Monte Carlo Algorithm")
    fmt.Println("=====================")

    MoCa.prediction()
    //time_taken := time.Since(MoCa.Time)
    //fmt.Println(MoCa, Result, "Time Taken", time_taken)
    fmt.Println(Result)
}

func (MoCa *MonteCarlo) convert_annual_values_to_monthly_values() {
    MoCa.MonthlyMeanReturn = MoCa.AnnualMeanReturn / 12
    MoCa.MonthlyRetStdDev = MoCa.AnnualReturnStandardDeviation / math.Sqrt(12)
    MoCa.MonthlyInflation = MoCa.AnnualInflation / 12
    MoCa.MonthlyInfStdDev = MoCa.AnnualInflationStandardDeviation / math.Sqrt(12)
}

func (MoCa *MonteCarlo) calculate_number_of_observations() {
    if MoCa.GraphType == "daily" {
        MoCa.NumberOfObservations = MoCa.Years * 365
    } else {
        MoCa.NumberOfObservations =  MoCa.Years * 12
    }
}

func (MoCa *MonteCarlo) simulate_monthly_invest_return() {
    random_norm_size := MoCa.NumberOfSimulations * MoCa.NumberOfObservations
    random_norm := generate_random_norm(random_norm_size, MoCa.MonthlyMeanReturn, MoCa.MonthlyRetStdDev)
    MoCa.MonthlyInvestmentReturns = make([][]float64, MoCa.NumberOfObservations)
    for i, j, k := 0, MoCa.NumberOfSimulations, 0; k < MoCa.NumberOfObservations; i += MoCa.NumberOfSimulations {
        MoCa.MonthlyInvestmentReturns[k] = make([]float64, MoCa.NumberOfSimulations)
        MoCa.MonthlyInvestmentReturns[k] = random_norm[i:j]
        k++
        j += MoCa.NumberOfSimulations
    }
    wg.Done()
}

func (MoCa *MonteCarlo) simulate_monthly_inflation_return() {
    random_norm_size := MoCa.NumberOfSimulations * MoCa.NumberOfObservations
    random_norm := generate_random_norm(random_norm_size, MoCa.MonthlyInflation, MoCa.MonthlyInfStdDev)
    MoCa.MonthlyInflationReturns = make([][]float64, MoCa.NumberOfObservations)
    for i, j, k := 0, MoCa.NumberOfSimulations, 0; k < MoCa.NumberOfObservations; i += MoCa.NumberOfSimulations {
        MoCa.MonthlyInflationReturns[k] = make([]float64, MoCa.NumberOfSimulations)
        MoCa.MonthlyInflationReturns[k] = random_norm[i:j]
        k++
        j += MoCa.NumberOfSimulations
    }
    wg.Done()
}

func generate_random_norm(n int, mean, std_dev float64) []float64 {
    random_norm := make([]float64, n)

    for i := 0; i < n; i++ {
        random_norm[i] = (rand.NormFloat64() * std_dev) + mean
    }

    return random_norm
}

func (MoCa *MonteCarlo) generate_combined_random_matrix() {
    MoCa.CombinedMatrix = make([][]float64, MoCa.NumberOfObservations)

    for i := 0 ; i < MoCa.NumberOfObservations ; i++ {
        MoCa.CombinedMatrix[i] = make([]float64, MoCa.NumberOfSimulations)
        for j := 0; j < MoCa.NumberOfSimulations ; j++ {
            MoCa.CombinedMatrix[i][j] = 1 + MoCa.MonthlyInvestmentReturns[i][j] - MoCa.MonthlyInflationReturns[i][j]
        }
    }
    wg.Done()
}

func (MoCa *MonteCarlo) generate_matrix_for_start_capital() {
    MoCa.StartCapitalMatrix = make([][]float64, MoCa.NumberOfObservations)
    for i := 0 ; i < MoCa.NumberOfObservations ; i++ {
        MoCa.StartCapitalMatrix[i] = make([]float64, MoCa.NumberOfSimulations)
        for j := 0; j < MoCa.NumberOfSimulations ; j++ {
            MoCa.StartCapitalMatrix[i][j] = MoCa.StartCapital
        }
    }
    wg.Done()
}

func (MoCa *MonteCarlo) simulate_nav_values_for_portfolio() {
    MoCa.NAVMatrix = MoCa.StartCapitalMatrix
    for i := 1; i < MoCa.NumberOfObservations; i++ {
        for j := 0; j < MoCa.NumberOfSimulations; j++ {
            MoCa.NAVMatrix[i][j] = (MoCa.NAVMatrix[i][j] * MoCa.CombinedMatrix[i][j]) - MoCa.MonthlyWithdrawals
        }
        sort.Float64s(MoCa.NAVMatrix[i])
    }
}

func average(xs[]float64)float64 {
    total:=0.0
    total_positive := 0
    for _,v:=range xs {
        if v > 0 {
            total +=v
            total_positive++
        }
    }
    return total/float64(total_positive)
}

func (MoCa *MonteCarlo) replace_all_values_less_than_zero_to_mean_of_row_in_nav_matrix() {
    for i := 1; i < MoCa.NumberOfObservations; i++ {
        for j := 0; j < MoCa.NumberOfSimulations; j++ {
            mean := average(MoCa.NAVMatrix[i])
            if MoCa.NAVMatrix[i][j] < 0 {
                MoCa.NAVMatrix[i][j] = mean
            }
        }
    }
    // Skip first row
    //MoCa.NAVMatrix = MoCa.NAVMatrix[1:]
}

//---------------------------------------------------------------------------
// Scenario Distribution
//---------------------------------------------------------------------------

func (MoCa *MonteCarlo) generate_scenario_distribution() {
    Result["good_scenario"] = make([][]interface{}, MoCa.NumberOfObservations)
    Result["average_scenario"] = make([][]interface{}, MoCa.NumberOfObservations)
    Result["bad_scenario"] = make([][]interface{}, MoCa.NumberOfObservations)
    for i := 0; i < MoCa.NumberOfObservations; i++ {
        Result["good_scenario"][i] = make([]interface{}, 2)
        Result["average_scenario"][i] = make([]interface{}, 2)
        Result["bad_scenario"][i] = make([]interface{}, 2)

        Result["good_scenario"][i][0] = int32(last_day_of_month(i).Unix())
        Result["average_scenario"][i][0] = int32(last_day_of_month(i).Unix())
        Result["bad_scenario"][i][0] = int32(last_day_of_month(i).Unix())

        Result["good_scenario"][i][1] = average(MoCa.NAVMatrix[i][int(float64(MoCa.NumberOfSimulations) * MoCa.GoodRange[0]):int(float64(MoCa.NumberOfSimulations) * MoCa.GoodRange[1])])
        Result["average_scenario"][i][1] = average(MoCa.NAVMatrix[i][int(float64(MoCa.NumberOfSimulations) * MoCa.AverageRange[0]):int(float64(MoCa.NumberOfSimulations) * MoCa.AverageRange[1])])
        Result["bad_scenario"][i][1] = average(MoCa.NAVMatrix[i][int(float64(MoCa.NumberOfSimulations) * MoCa.BadRange[0]):int(float64(MoCa.NumberOfSimulations) * MoCa.BadRange[1])])
    }
}

func last_day_of_month(inc_month int) time.Time {
    ist, _ := time.LoadLocation("Asia/Kolkata")
    now := time.Now().In(ist)
    currentYear, currentMonth, _ := now.Date()
    currentLocation := now.Location()
    
    newMonth := int(currentMonth) + inc_month
    // Date(year int, month Month, day, hour, min, sec, nsec int, loc *Location) Time
    firstOfMonth := time.Date(currentYear, time.Month(newMonth), 1, 0, 0, 0, 0, currentLocation)
    return firstOfMonth.AddDate(0, 1, -1)
}
