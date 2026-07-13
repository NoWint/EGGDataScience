"""
被试管理 API 路由
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.database import (
    create_subject, get_subject, list_subjects,
    update_subject, delete_subject,
    create_experiment, list_experiments, delete_experiment,
)

router = APIRouter(prefix="/api/subjects", tags=["subjects"])


# ========== 请求模型 ==========
class SubjectCreate(BaseModel):
    code: str
    age: Optional[int] = None
    gender: Optional[str] = None


class SubjectUpdate(BaseModel):
    code: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None


class ExperimentCreate(BaseModel):
    subject_id: int
    condition: str
    date: Optional[str] = None
    notes: Optional[str] = None


# ========== 被试管理 ==========
@router.get("")
@router.get("/")
async def get_subjects():
    """列出所有被试"""
    return list_subjects()


@router.post("")
@router.post("/")
async def create_subject_api(req: SubjectCreate):
    """创建被试"""
    try:
        return create_subject(req.code, req.age, req.gender)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/{subject_id}")
async def get_subject_api(subject_id: int):
    """获取单个被试"""
    subject = get_subject(subject_id)
    if not subject:
        raise HTTPException(404, f"未找到被试: {subject_id}")
    return subject


@router.put("/{subject_id}")
async def update_subject_api(subject_id: int, req: SubjectUpdate):
    """更新被试"""
    try:
        subject = update_subject(subject_id, req.code, req.age, req.gender)
        if not subject:
            raise HTTPException(404, f"未找到被试: {subject_id}")
        return subject
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/{subject_id}")
async def delete_subject_api(subject_id: int):
    """删除被试"""
    if not delete_subject(subject_id):
        raise HTTPException(404, f"未找到被试: {subject_id}")
    return {"status": "deleted", "id": subject_id}


# ========== 被试的实验记录 ==========
@router.get("/{subject_id}/experiments")
async def get_subject_experiments(subject_id: int):
    """列出被试的实验记录"""
    if not get_subject(subject_id):
        raise HTTPException(404, f"未找到被试: {subject_id}")
    return list_experiments(subject_id=subject_id)


@router.post("/{subject_id}/experiments")
async def create_experiment_api(subject_id: int, req: ExperimentCreate):
    """为被试创建实验记录"""
    if not get_subject(subject_id):
        raise HTTPException(404, f"未找到被试: {subject_id}")
    return create_experiment(req.subject_id, req.condition, req.date, req.notes)


# ========== 实验记录管理 ==========
@router.get("/experiments/all")
async def get_all_experiments():
    """列出所有实验记录"""
    return list_experiments()


@router.delete("/experiments/{exp_id}")
async def delete_experiment_api(exp_id: int):
    """删除实验记录"""
    if not delete_experiment(exp_id):
        raise HTTPException(404, f"未找到实验记录: {exp_id}")
    return {"status": "deleted", "id": exp_id}
