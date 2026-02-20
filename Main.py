# Fastapi_models
def require_role(role: str):
    def checker(current_user: User = Depends(get_current_active_user)):
        if current_user.role != role:
            raise HTTPException(status_code=403, detail="Forbidden")
        return current_user
    return checker

@app.delete("/admin/users/{id}")
async def delete_user(user=Depends(require_role("admin"))):
    ...
