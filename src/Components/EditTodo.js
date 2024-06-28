import React, { useState } from 'react'

const EditTodo = ({editTodo,task}) => {
    const [value,Setvalue] = useState(task.task)
    const handleChange = (e) => {
        e.preventDefault()
        editTodo(value,task.id)
        Setvalue("")

    }
  return (
    <form onSubmit={handleChange} className='TodoForm'>
        <input type='text' className='todo-input' placeholder='Update task' value={value} onChange={(e) => {
            Setvalue(e.target.value)
        }}/>
        <button type='submit' className='todo-btn'>Update task</button>
    </form>
  )
}

export default EditTodo
