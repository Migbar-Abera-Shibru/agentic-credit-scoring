# demo.py
import requests
import json

def interactive_demo():
    print("🎪 Agentic Credit Scoring Interactive Demo")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    while True:
        print("\n📋 Choose an option:")
        print("1. Test pre-defined scenarios")
        print("2. Enter custom applicant data")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            test_predefined_scenarios(base_url)
        elif choice == "2":
            test_custom_applicant(base_url)
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

def test_predefined_scenarios(base_url):
    scenarios = [
        {
            "name": "Ideal Applicant",
            "features": {
                "RevolvingUtilizationOfUnsecuredLines": 0.1, "age": 45, 
                "NumberOfTime30-59DaysPastDueNotWorse": 0, "DebtRatio": 0.15,
                "MonthlyIncome": 10000, "NumberOfOpenCreditLinesAndLoans": 4,
                "NumberOfTimes90DaysLate": 0, "NumberRealEstateLoansOrLines": 2,
                "NumberOfTime60-89DaysPastDueNotWorse": 0, "NumberOfDependents": 2
            }
        },
        {
            "name": "Risky Applicant", 
            "features": {
                "RevolvingUtilizationOfUnsecuredLines": 0.95, "age": 22,
                "NumberOfTime30-59DaysPastDueNotWorse": 5, "DebtRatio": 0.8,
                "MonthlyIncome": 1500, "NumberOfOpenCreditLinesAndLoans": 12,
                "NumberOfTimes90DaysLate": 2, "NumberRealEstateLoansOrLines": 0,
                "NumberOfTime60-89DaysPastDueNotWorse": 1, "NumberOfDependents": 1
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🎯 Testing: {scenario['name']}")
        test_applicant(base_url, scenario["features"])

def test_custom_applicant(base_url):
    print("\n👤 Enter applicant details:")
    
    features = {}
    features["RevolvingUtilizationOfUnsecuredLines"] = float(input("Credit Utilization (0-1): "))
    features["age"] = int(input("Age: "))
    features["NumberOfTime30-59DaysPastDueNotWorse"] = int(input("30-59 Days Late Payments: "))
    features["DebtRatio"] = float(input("Debt Ratio: "))
    features["MonthlyIncome"] = float(input("Monthly Income: "))
    features["NumberOfOpenCreditLinesAndLoans"] = int(input("Open Credit Lines: "))
    features["NumberOfTimes90DaysLate"] = int(input("90+ Days Late Payments: "))
    features["NumberRealEstateLoansOrLines"] = int(input("Real Estate Loans: "))
    features["NumberOfTime60-89DaysPastDueNotWorse"] = int(input("60-89 Days Late Payments: "))
    features["NumberOfDependents"] = int(input("Dependents: "))
    
    test_applicant(base_url, features)

def test_applicant(base_url, features):
    try:
        # Get prediction
        predict_response = requests.post(f"{base_url}/predict", json={"features": features})
        prediction = predict_response.json()
        
        print(f"\n📊 PREDICTION RESULTS:")
        print(f"Probability of Delinquency: {prediction['probability']:.1%}")
        
        # Get agent analysis
        agent_response = requests.post(f"{base_url}/agent", json={
            "features": features,
            "query": "Explain all risk factors and provide recommendations"
        })
        
        agent_result = agent_response.json()
        
        print(f"Risk Level: {agent_result['risk_level']}")
        print(f"\n🔍 RISK FACTORS:")
        for factor in agent_result['risk_factors']:
            print(f"  • {factor}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for recommendation in agent_result['recommendations']:
            print(f"  • {recommendation}")
            
        # Test what-if scenario
        whatif_response = requests.post(f"{base_url}/agent", json={
            "features": features,
            "query": "What if I reduce my credit utilization by 50%?"
        })
        
        if whatif_response.status_code == 200:
            whatif_result = whatif_response.json()
            print(f"\n🔮 WHAT-IF SCENARIO:")
            for reasoning in whatif_result['reasoning'][-1:]:  # Last reasoning line
                if "Scenario simulation" in reasoning:
                    print(f"  {reasoning}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_demo()