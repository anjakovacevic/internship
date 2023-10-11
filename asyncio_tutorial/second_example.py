import api
import asyncio

async def kill_time(num):
    print("Running: ", num)
    await asyncio.sleep(1)
    print('Finished: ', num)


async def main():
    print("Started")
    list_of_tasks = []
    for i in range(1, 1000+1):
        list_of_tasks.append(kill_time(i))
    
    await asyncio.sleep(2)
    await asyncio.gather(*list_of_tasks)    # running so many functions at the same time
    print("Done")

if __name__ == '__main__':
    asyncio.run(main())