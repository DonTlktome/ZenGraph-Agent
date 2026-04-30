"""
验证 #1 修复：retrieve_node 和 grader_node 正确消费 standalone_query。

Bug 背景：
  contextualize_node 和 hyde_node 生成的 standalone_query 在进入 retrieve_node 和
  grader_node 时被丢弃，始终用原始 query 检索和评估，两条增强路径形同虚设。

  - Bug #1: retrieve_node 硬编码 state["query"]
  - Bug #2: grader_node 硬编码 state["query"]
  - 修复: 改为 state.get("standalone_query") or state["query"]
"""

import sys
import os

# 确保项目根目录在 sys.path 中（test 文件在 tests/ 子目录下）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
from src.schema import AgentState


# ---------------------------------------------------------------------------
# 辅助函数：构造一个最小有效的 AgentState
# ---------------------------------------------------------------------------
def _make_state(query: str = "", standalone_query: str = "", **overrides) -> AgentState:
    """构造 AgentState，只填必填字段，其他用默认值。"""
    base: AgentState = {
        "query": query,
        "standalone_query": standalone_query,
        "route": "",
        "retrieved_context": "",
        "final_answer": "",
        "grade": "",
        "loop_step": 0,
        "chat_history": [],
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


# ---------------------------------------------------------------------------
# retrieve_node 测试
# ---------------------------------------------------------------------------
class TestRetrieveNode:
    """验证 retrieve_node 的数据读取优先级"""

    def test_uses_standalone_query_when_present(self):
        """核心断言：有 standalone_query 时必须用它检索，而非原始 query"""
        from src.nodes import retrieve_node

        state = _make_state(
            query="那具体该怎么做呢？",
            standalone_query="如何克服焦虑？",
            route="contextualize",
            chat_history=["信众: 我很焦虑", "法师: 阿弥陀佛..."],
        )

        with patch("src.nodes.retriever_obj") as mock_retriever:
            mock_retriever.query.return_value = "焦虑对治经文..."
            retrieve_node(state)

        mock_retriever.query.assert_called_once_with("如何克服焦虑？")

    def test_falls_back_to_query_when_standalone_is_empty_string(self):
        """standalone_query 为空字符串（falsy），应回退到原始 query"""
        from src.nodes import retrieve_node

        state = _make_state(
            query="什么是空？",
            standalone_query="",
            route="direct",
        )

        with patch("src.nodes.retriever_obj") as mock_retriever:
            mock_retriever.query.return_value = "空性经文..."
            retrieve_node(state)

        mock_retriever.query.assert_called_once_with("什么是空？")

    def test_falls_back_to_query_when_standalone_not_set(self):
        """standalone_query 为 TypedDict 默认空值时的回退"""
        from src.nodes import retrieve_node

        # 模拟 TypedDict 创建后未显式设置 standalone_query 的场景
        state: AgentState = {
            "query": "什么是空？",
            "standalone_query": "",
            "route": "",
            "retrieved_context": "",
            "final_answer": "",
            "grade": "",
            "loop_step": 0,
            "chat_history": [],
        }

        with patch("src.nodes.retriever_obj") as mock_retriever:
            mock_retriever.query.return_value = "..."
            retrieve_node(state)

        mock_retriever.query.assert_called_once_with("什么是空？")

    def test_uses_standalone_query_from_hyde_path(self):
        """HyDE 路径：假设性回答作为 search_query"""
        from src.nodes import retrieve_node

        hyde_answer = "一切法无自性，了不可得，如梦幻泡影..."
        state = _make_state(
            query="空是什么？",
            standalone_query=hyde_answer,
            route="hyde",
        )

        with patch("src.nodes.retriever_obj") as mock_retriever:
            mock_retriever.query.return_value = "..."
            retrieve_node(state)

        mock_retriever.query.assert_called_once_with(hyde_answer)


# ---------------------------------------------------------------------------
# grader_node 测试
# ---------------------------------------------------------------------------
class TestGraderNode:
    """验证 grader_node 用增强查询做相关性评估"""

    def test_evaluates_with_standalone_query(self):
        """
        Grader 的 prompt 中必须包含增强后的查询。
        这是 HyDE 路径的关键：检索回来的经文是跟假设性回答匹配的，
        用原始 query 评估会误判为不相关。
        """
        from src.nodes import grader_node

        state = _make_state(
            query="那具体该怎么做呢？",
            standalone_query="如何克服焦虑？",
            retrieved_context="应观法界性，一切唯心造...",
        )

        with patch("src.nodes.get_deepseek_model") as mock_get_model:
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "yes"
            mock_model.run.return_value = mock_response
            mock_get_model.return_value = mock_model

            grader_node(state)

        # 提取传给模型的 prompt 文本
        call_args = mock_model.run.call_args[0][0]
        prompt_text = call_args[0]["content"]

        assert "如何克服焦虑？" in prompt_text, (
            "Grader prompt 应包含增强后的查询，实际 prompt:\n" + prompt_text
        )

    def test_falls_back_to_original_query(self):
        """无 standalone_query 时 prompt 中应有原始 query"""
        from src.nodes import grader_node

        state = _make_state(
            query="什么是空？",
            standalone_query="",
            retrieved_context="色不异空...",
        )

        with patch("src.nodes.get_deepseek_model") as mock_get_model:
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "yes"
            mock_model.run.return_value = mock_response
            mock_get_model.return_value = mock_model

            grader_node(state)

        call_args = mock_model.run.call_args[0][0]
        prompt_text = call_args[0]["content"]
        assert "什么是空？" in prompt_text
