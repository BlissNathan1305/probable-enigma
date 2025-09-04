leaders = [
    {"name": "Nayib Bukele", "country": "El Salvador", "approval_rating": 91},
    {"name": "Ilham Aliyev", "country": "Azerbaijan", "approval_rating": 91},
    {"name": "Assimi Goïta", "country": "Mali", "approval_rating": 91.8},
    {"name": "Vladimir Putin", "country": "Russia", "approval_rating": 87},
    {"name": "Ibrahim Traoré", "country": "Burkina Faso", "approval_rating": 87.6},
    {"name": "Narendra Modi", "country": "India", "approval_rating": 73},
]

# Sort leaders by approval rating
leaders.sort(key=lambda x: x["approval_rating"], reverse=True)

# Print the list of leaders
for leader in leaders:
    print(f"{leader['name']} ({leader['country']}) - Approval Rating: {leader['approval_rating']}%")
