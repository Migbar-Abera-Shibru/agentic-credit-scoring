# final_demonstration.py
import requests
import json

def final_demonstration():
    print(" FINAL AGENTIC CREDIT SCORING DEMONSTRATION")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Test cases that show different risk levels
    test_cases = [
        {
            "name": " VERY HIGH RISK APPLICANT",
            "features": {
                "RevolvingUtilizationOfUnsecuredLines": 1.2,  # Over 100% utilization
                "age": 21,
                "NumberOfTime30-59DaysPastDueNotWorse": 8,
                "DebtRatio": 1.1,
                "MonthlyIncome": 1200,
                "NumberOfOpenCreditLinesAndLoans": 15,
                "NumberOfTimes90DaysLate": 3,
                "NumberRealEstateLoansOrLines": 0,
                "NumberOfTime60-89DaysPastDueNotWorse": 2,
                "NumberOfDependents": 3
            }
        },
        {
            "name": " IDEAL APPLICANT", 
            "features": {
                "RevolvingUtilizationOfUnsecuredLines": 0.1,
                "age": 45,
                "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.15,
                "MonthlyIncome": 12000,
                "NumberOfOpenCreditLinesAndLoans": 4,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 2,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}")
        print("-" * 40)
        
        try:
            # Get full agentic analysis
            response = requests.post(f"{base_url}/agent", json={
                "features": test_case["features"],
                "query": "Provide a complete risk analysis with specific recommendations"
            })
            
            if response.status_code == 200:
                result = response.json()
                
                print(f" Probability: {result['probability']:.1%}")
                print(f" Risk Level: {result['risk_level']}")
                
                print(f"\n🔍 REASONING:")
                for reasoning in result['reasoning'][:4]:  # Show first 4 reasoning steps
                    print(f"   {reasoning}")
                
                if result['risk_factors']:
                    print(f"\n  RISK FACTORS:")
                    for factor in result['risk_factors']:
                        print(f"   • {factor}")
                
                if result['recommendations']:
                    print(f"\n RECOMMENDATIONS:")
                    for rec in result['recommendations']:
                        print(f"   • {rec}")
                
                print(f"\n🛠️  AGENT TOOLS USED: {', '.join(result['tools_used'])}")
                
            else:
                print(f"❌ Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")

    print(f"\n{'='*50}")
    print("🎉 DEMONSTRATION COMPLETE!")
    print("🤖 Your Agentic Credit Scoring System is fully operational!")
    print("🌐 Access the API at: http://localhost:8000/docs")

if __name__ == "__main__":
    final_demonstration()


   