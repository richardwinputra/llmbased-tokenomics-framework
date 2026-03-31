import json

def generate_cases():
    cases = [
        {
            "input_type": "structured",
            "project_name": "DeFiLend",
            "token_symbol": "LEND",
            "core_principles": ["Decentralization", "Capital Efficiency", "Risk Management"],
            "token_purpose": ["Governance", "Fee sharing", "Staking"],
            "token_functions": ["Vote on risk parameters", "Earn protocol revenue", "Stake to insure protocol"],
            "total_supply_preference": "Fixed 1B",
            "inflation_preference": "None",
            "economic_design": "Value accrual via buy and burn from protocol fees.",
            "legal_design": "Utility token, not a security",
            "technical_design": "ERC-20 on Ethereum L1",
            "gov_structures": "DAO with token voting",
            "stakeholders": ["Liquidity Providers", "Borrowers", "Stakers", "Core Team"],
            "similar_projects": ["Aave", "Compound"]
        },
        {
            "input_type": "structured",
            "project_name": "GameFiWorld",
            "token_symbol": "GFW",
            "core_principles": ["Play to Earn", "Community Ownership"],
            "token_purpose": ["In-game currency", "Governance"],
            "token_functions": ["Buy items", "Breed NFTs", "Vote on game updates"],
            "total_supply_preference": "Dynamic, capped at 10B",
            "inflation_preference": "Inflationary to reward players",
            "economic_design": "Sink mechanics via breeding and upgrading items.",
            "legal_design": "In-game currency",
            "technical_design": "ERC-20 on Polygon L2",
            "gov_structures": "Foundation transitioning to DAO",
            "stakeholders": ["Players", "Guilds", "Investors", "Game Studio"],
            "similar_projects": ["Axie Infinity", "Illuvium"]
        },
        {
            "input_type": "structured",
            "project_name": "DePINStorage",
            "token_symbol": "STOR",
            "core_principles": ["Decentralized Storage", "Censorship Resistance"],
            "token_purpose": ["Payment for storage", "Reward for miners"],
            "token_functions": ["Pay for file hosting", "Provide collateral to mine"],
            "total_supply_preference": "Capped at 2B",
            "inflation_preference": "Algorithmic decay emission",
            "economic_design": "Burn and mint equilibrium mechanism.",
            "legal_design": "Utility",
            "technical_design": "Native L1",
            "gov_structures": "Protocol governance",
            "stakeholders": ["Storage Providers", "Users", "Developers"],
            "similar_projects": ["Filecoin", "Arweave"]
        }
    ]
    
    with open("sample_batch_inputs.json", "w") as f:
        json.dump(cases, f, indent=4)
        
    print("Created sample_batch_inputs.json with 3 diverse test cases.")

if __name__ == "__main__":
    generate_cases()
